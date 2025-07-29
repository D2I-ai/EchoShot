"""
Diffusion processes each wraps denoising, diffusion, sampling, and loss functions.
"""
import torch
import numpy as np

from utils.fm_solvers import FlowDPMSolverMultistepScheduler
# from .schema import TensorList, randn_like
from utils.utils import (
    randn_like,
    randn_like,
    TensorList
)
import copy
from tqdm import tqdm

__all__ = ['DiffusionProcess']

def retrieve_timesteps(
    scheduler,
    num_inference_steps= None,
    device= None,
    timesteps= None,
    sigmas = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps+1)[:sampling_steps]
    sigma = (shift * sigma / (1 + (shift - 1) * sigma))

    return sigma

class DiffusionProcess:

    def __init__(self, schedule, prediction_type='eps'):
        assert isinstance(schedule, FlowDPMSolverMultistepScheduler)
        assert prediction_type in ('flow_prediction')
        self.schedule = schedule
        self.schedule_copy = copy.deepcopy(self.schedule)
        
        self.prediction_type = prediction_type

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.schedule_copy.sigmas.to(device=timesteps.device, dtype=dtype)
        schedule_timesteps = self.schedule_copy.timesteps.to(timesteps.device)
        # timesteps = timesteps.to(timesteps.device)
        # step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        step_indice = self.schedule_copy.index_for_timestep(timesteps)

        sigma = sigmas[step_indice].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def diffuse(self, x0, t, noise=None, generator=None):
        """
        Add Gaussian noise to signal x0.
        """
        # bsz = x0.size(0)
        # noise = randn_like(x0, generator=generator) if noise is None else noise
        # u = compute_density_for_timestep_sampling(
        #     weighting_scheme='logit_normal',
        #     batch_size=bsz,
        #     logit_mean=0.0,
        #     logit_std=1.0,
        #     mode_scale=1.29, # Only effective when using the `'mode'` as the `weighting_scheme`
        # )
        # indices = (u * self.schedule_copy.num_train_timesteps).long()
        # timesteps = self.schedule_copy.timesteps[indices].to(device=x0.device)

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        # sigmas = self.get_sigmas(timesteps, n_dim=x0.ndim, dtype=x0.dtype)
        # xt = self.schedule.add_noise(x0, noise=noise, timesteps=t)
        # xt = (1.0 - sigmas) * x0 + sigmas * noise

        ### new branch
        noise = randn_like(x0, generator=generator) if noise is None else noise
        xt = self.schedule.add_noise(x0, noise=noise, timesteps=t)
        return xt
    
    def denoise(
        self,
        xt,
        t,
        model,
        model_kwargs={},
        guide_scale=None,
        guide_rescale=None,
        clamp=None,
        percentile=None
    ):
        """
        Apply one step of denoising on xt to get x0.
        """
        if isinstance(xt, list):
            xt = TensorList(xt)
        assert isinstance(xt, (torch.Tensor, TensorList))

        # wrap the model to support TensorList
        def model_fn(*args, **kwargs):
            out = model(*args, **kwargs)
            if isinstance(out, list):
                out = TensorList(out)
            return out

        # model forward
        if guide_scale is None:
            assert isinstance(model_kwargs, dict)
            out = model_fn(xt, t=t, **model_kwargs)
        else:
            # classifier-free guidance (arXiv:2207.12598)
            # model_kwargs[0]: conditional kwargs
            # model_kwargs[1]: non-conditional kwargs
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
            y_out = model_fn(xt, t=t, **model_kwargs[0])
            if guide_scale == 1.:
                out = y_out
            else:
                u_out = model_fn(xt, t=t, **model_kwargs[1])
                out = u_out + guide_scale * (y_out - u_out)

                # rescale the output according to arXiv:2305.08891
                if guide_rescale is not None:
                    assert guide_rescale >= 0 and guide_rescale <= 1
                    ratio = (torch.stack([u.std() for u in y_out]) / (
                        torch.stack([u.std() for u in out]) + 1e-12
                    )).view((-1, ) + (1, ) * (y_out.ndim - 1))
                    out *= guide_rescale * ratio + (1 - guide_rescale) * 1.0
        
        # calculate x0
        x0 = out
        
        # restrict the range of x0
        if percentile is not None:
            # NOTE: percentile should only be used when data is within range [-1, 1]
            assert percentile > 0 and percentile <= 1
            k = torch.quantile(x0.flatten(1).abs(), percentile, dim=1)
            k = k.clamp_(1.0).view((-1, ) + (1, ) * (xt.ndim - 1))
            x0 = torch.min(k, torch.max(-k, x0)) / k
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)
        return x0
    
    def loss(
        self,
        x0,
        t,
        model,
        model_kwargs={},
        min_snr_gamma=None,
        generator=None
    ):
        if isinstance(x0, list):
            x0 = TensorList(x0)
        assert isinstance(x0, (torch.Tensor, TensorList))

        # diffuse
        noise = randn_like(x0, generator=generator)
        xt = self.diffuse(x0, t, noise)

        # denoise
        out = model(xt, t=t, **model_kwargs)
        if isinstance(out, list):
            out = TensorList(out)
        
        target = noise - x0

        # mse loss
        loss = (out - target) ** 2
        loss = torch.stack([u.mean() for u in loss])

        # min-snr reweighting
        if min_snr_gamma is not None:
            # hyperparams
            sigma = self.get_sigmas(t, n_dim=x0.ndim, dtype=x0.dtype)
            alpha = (1 - sigma ** 2) ** 0.5
            snr = torch.pow(alpha / sigma, 2).clamp_(1e-20)
            min_snr = snr.clamp(max=min_snr_gamma)
            loss = loss * (min_snr / snr)
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        noise,
        model,
        model_kwargs={},
        steps=20,
        sample_shift=5.0,
        training_steps=1000,
        base_seed=0,
        sample_guide_scale=5.0,
        **kwargs
    ):
        sample_scheduler = FlowDPMSolverMultistepScheduler(num_train_timesteps=training_steps, shift=5.0, use_dynamic_shifting=False)
        
        sampling_sigmas = get_sampling_sigmas(steps, sample_shift)
        

        # sample videos
        latents = noise
        device = latents[0].device

        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(base_seed)

        for j in range(len(latents)):

            timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)
            arg_c = {'context': model_kwargs[0]['context'][j:j+1], 'seq_len': model_kwargs[0]['seq_len']}
            arg_null = {'context': model_kwargs[1]['context'][j:j+1], 'seq_len': model_kwargs[0]['seq_len']}

            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents[j:j+1]
                timestep = [t]

                timestep = torch.stack(timestep)

                noise_pred_cond = TensorList(model(TensorList(latent_model_input), t=timestep, **arg_c))
                noise_pred_uncond = TensorList(model(TensorList(latent_model_input), t=timestep, **arg_null))

                noise_pred = noise_pred_uncond + sample_guide_scale * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(noise_pred[0].unsqueeze(0), t, latents[j].unsqueeze(0), return_dict=False, generator=seed_g)[0]
                latents[j] = temp_x0.squeeze(0)

            x0 = latents
        return x0

