import torch
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from functools import partial
from contextlib import nullcontext

from utils import HuggingfaceTokenizer

from utils.fm_solvers import FlowDPMSolverMultistepScheduler
from .diffusion import DiffusionProcess

import models
from models.vae import VideoVAE
from models.model import Transformer

from utils.utils import (
    to_
)

__all__ = ['WanxgenMulshot']


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True 
):
    if dist.is_initialized():
        model = FSDP(
            module=model,
            process_group=None,
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda m: m in model.blocks
            ),
            mixed_precision=MixedPrecision(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                buffer_dtype=buffer_dtype
            ),
            device_id=device_id,
            sync_module_states=sync_module_states
        )
    else:
        model = model.to(dtype=param_dtype, device=device_id)
    return model


class T5Encoder:

    def __init__(
        self,
        name,
        text_len,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        checkpoint_path=None,
        tokenizer_path=None,
        shard_fn=None
    ):
        self.name = name
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        model = shard_fn(getattr(models, name)(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device=device
        ).eval().requires_grad_(False))
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model = model

        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path,
            seq_len=text_len,
            clean='whitespace'
        )
    
    def __call__(self, texts):
        ids, mask = to_(self.tokenizer(
            texts,
            return_mask=True,
            add_special_tokens=True
        ), self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]


class WanxgenMulshot:
    """
    Wraps modules (autoencoder, text encoder, and diffusion transformer) and utilities
    (sampling, etc.) of Minisora.
    """
    def __init__(
        self,
        device_id,
        fsdp_param_dtype=torch.bfloat16,
        fsdp_reduce_dtype=torch.float32,
        fsdp_buffer_dtype=torch.float32,
        fsdp_sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        cfg=None,
    ):
        # sharding function
        shard_fn = partial(
            shard_model,
            device_id=device_id,
            param_dtype=fsdp_param_dtype,
            reduce_dtype=fsdp_reduce_dtype,
            buffer_dtype=fsdp_buffer_dtype,
            sharding_strategy=fsdp_sharding_strategy
        )
        
        device = f'cuda:{device_id}'

        # [model] t5
        self.t5 = T5Encoder(
            name=cfg.t5_model,
            text_len=cfg.text_len,
            dtype=cfg.t5_dtype,
            device=device,
            checkpoint_path=cfg.t5_checkpoint,
            tokenizer_path=cfg.t5_tokenizer,
            shard_fn=shard_fn
        )


        # [model] vae
        self.vae = VideoVAE(
            vae_pth=cfg.vae_checkpoint,
            device=f'cuda:{device_id}'
        )

        # [model] transformer
        model = Transformer(
            patch_size=cfg.patch_size,
            text_len=cfg.text_len,
            in_dim=self.vae.model.z_dim,
            dim=cfg.dim,
            ffn_dim=cfg.ffn_dim,
            freq_dim=cfg.freq_dim,
            text_dim=self.t5.model.dim,
            out_dim=self.vae.model.z_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            window_size=cfg.window_size,
            qk_norm=cfg.qk_norm,
            cross_attn_norm=cfg.cross_attn_norm,
            eps=cfg.eps,
            num_grad_checkpoints=cfg.num_grad_checkpoints
        )

        checkpoint_path = getattr(cfg, 'checkpoint_path', None)
        if checkpoint_path:
            state_dict = torch.load(cfg.checkpoint_path, map_location='cpu')
            status = model.load_state_dict(state_dict, strict=True)
        self.dit = shard_fn(model)

        # [model] transformer EMA
        if cfg.use_ema:
            dit_ema = Transformer(
                patch_size=cfg.patch_size,
                text_len=cfg.text_len,
                in_dim=self.vae.model.z_dim,
                dim=cfg.dim,
                ffn_dim=cfg.ffn_dim,
                freq_dim=cfg.freq_dim,
                text_dim=self.t5.model.dim,
                out_dim=self.vae.model.z_dim,
                num_heads=cfg.num_heads,
                num_layers=cfg.num_layers,
                window_size=cfg.window_size,
                qk_norm=cfg.qk_norm,
                cross_attn_norm=cfg.cross_attn_norm,
                eps=cfg.eps,
                num_grad_checkpoints=cfg.num_grad_checkpoints
            )
            checkpoint_ema_path = getattr(cfg, 'checkpoint_ema_path', None)
            if checkpoint_ema_path:
                dit_ema.load_state_dict(torch.load(
                    cfg.checkpoint_ema_path, map_location='cpu'
                ))
            self.dit_ema = shard_fn(dit_ema)
        else:
            self.dit_ema = None


        self.schedule = FlowDPMSolverMultistepScheduler(num_train_timesteps=cfg.num_train_timesteps, shift=5.0, use_dynamic_shifting=False)

        self.diffusion = DiffusionProcess(
            schedule=self.schedule,
            prediction_type="flow_prediction"
        )

        self.dtype = fsdp_param_dtype
        self.device = torch.device(device_id)
    
    @torch.no_grad()
    def sample(
        self,
        prompt,
        neg_prompt='',
        width=1280,
        height=720,
        num_frames=None,  # if None, auto-deduce
        fps=16,
        guide_scale=5.0,
        guide_rescale=0.5,
        solver='dpmpp_2m_sde',
        schedule=None,
        steps=50,
        seed=2024,
        show_progress=False
    ):
        # sanity check
        assert height % self.downsample[1] == 0
        assert width % self.downsample[2] == 0
        if num_frames is None:
            latent_area = (height // self.downsample[1]) * (width // self.downsample[2])
            num_frames = (self.dit.seq_len // latent_area - 1) * self.downsample[0] + 1
        assert (num_frames - 1) % self.downsample[0] == 0
        assert (
            ((num_frames - 1) // self.downsample[0] + 1) *
            (height // self.downsample[1]) *
            (width // self.downsample[2])
        ) <= self.dit.seq_len

        # init conditions
        pos_context = self.t5([prompt])
        neg_context = self.t5([neg_prompt])
        
        # init noise
        g = torch.Generator(device=self.device)
        noise = torch.randn(
            size=(
                1,
                self.dae.model.z_dim,
                (num_frames - 1) // self.dae_stride[0] + 1,
                height // self.dae_stride[1],
                width // self.dae_stride[2]
            ),
            generator=g,
            device=self.device
        )

        # sample video
        with amp.autocast(dtype=self.dtype), (
            self.dit.no_sync() if dist.is_initialized() else nullcontext()
        ):
            # sample latent video
            z = self.diffusion.sample(
                noise=noise,
                model=self.dit,
                model_kwargs=[
                    {'context': pos_context},   # positive condition
                    {'context': neg_context}    # negative condition
                ],
                guide_scale=guide_scale,
                guide_rescale=guide_rescale,
                solver=solver,
                schedule=schedule,
                steps=steps,
                show_progress=show_progress,
                seed=seed
            )
        
        # decode video
        video = self.dae.decode(z)
        return torch.stack(video)