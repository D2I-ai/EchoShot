import os
import os.path as osp
import sys
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.cuda.amp as amp
import torch.optim as optim
import json
import numpy as np
import logging
import datetime
import time
import decord
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from diffusers.training_utils import compute_density_for_timestep_sampling
from importlib import reload
from collections import OrderedDict
from decord import VideoReader

from utils.utils import to_

from models.wanxgen import WanxgenMulshot

import random
import shutil, glob

__all__ = ['main']

def time2frame(time):
    return [(t-1)*4+1 for t in time]

class VideoFolder(Dataset):

    def __init__(
        self,
        list_file,
        seq_len=61216,
        downsample=(4, 16, 16),
        min_area=round((1280 * 720) * 0.8),
        max_area=round((1280 * 720) / 0.8),
        fps=16,
        seed=2024,
        cfg=None,
    ):
        assert min_area <= max_area
        assert seq_len >= min_area / (downsample[1] * downsample[2])
        self.list_file = list_file
        self.seq_len = seq_len
        self.downsample = downsample
        self.min_area = min_area
        self.max_area = max_area
        self.fps = fps
        self.seed = seed
        self.cfg = cfg

        self.shuffle = False
        # read items
        with open(list_file) as f:
            self.items = json.load(f)
        
        if self.shuffle:
            random.shuffle(self.items)

    def __getitem__(self, index):
        while(True):
            try:
                return self.get_item(index)
            except Exception as e:
                print(e)
                path = self.items[index]['shots']
                print(f'Error when loading {path}. Trying anothre one.')
                index = random.randint(0, len(self.items) - 1)
                continue

    def get_item(self, index):
        # parse item
        item = self.items[index]
        paths, caption = item['shots'], item['cap_list']
        inner_t = item['inner_t']

        # seeding (need to reseed for each thread under multi-threading)
        rng = np.random.default_rng(self.seed + hash(paths[0]) % 10000)

        # read video and parse meta information
        decord.bridge.set_bridge('torch')

        video_list = []
        for i, path in enumerate(paths):
            reader = VideoReader(path)
            fps = reader.get_avg_fps()
            frame_timestamps = np.array(
                [reader.get_frame_timestamp(i) for i in range(len(reader))],
                dtype=np.float32
            )
            duration = frame_timestamps[-1].mean()
            df, dh, dw = self.downsample

            of = inner_t[i]

            of = (of - 1) * df + 1
            oh = self.cfg.H
            ow = self.cfg.W

            # sample frame ids
            target_duration = of / self.fps
            begin = rng.uniform(0, duration - target_duration)
            timestamps = np.linspace(begin, begin + target_duration, of)
            frame_ids = np.argmax(np.logical_and(
                timestamps[:, None] >= frame_timestamps[None, :, 0],
                timestamps[:, None] < frame_timestamps[None, :, 1]
            ), axis=1).tolist()

            # preprocess video
            video = reader.get_batch(frame_ids)             # [t, h, w, c]  
            video = self._preprocess_video(video, oh, ow)   # [c, t, h, w]
            
            video_list.append(video)

        videos = torch.cat(video_list, dim=1)    # [c, t_sum, h, w]
        return videos, caption, inner_t
    
    def __len__(self):
        return len(self.items)
    
    def _preprocess_video(self, video, oh, ow):
        """
        Resize, center crop, convert to tensor, and normalize.
        """
        # permute ([t, h, w, c] -> [t, c, h, w])
        video = video.permute(0, 3, 1, 2)

        # resize and crop
        ih, iw = video.shape[2:]
        if ih != oh or iw != ow:
            # resize
            scale = max(ow / iw, oh / ih)
            video = F.interpolate(
                video,
                size=(round(scale * ih), round(scale * iw)),
                mode='bicubic',
                antialias=True
            )
            assert video.size(3) >= ow and video.size(2) >= oh

            # center crop
            x1 = (video.size(3) - ow) // 2
            y1 = (video.size(2) - oh) // 2
            video = video[:, :, y1:y1 + oh, x1:x1 + ow]

        # permute ([t, c, h, w] -> [c, t, h, w]) and normalize
        video = video.transpose(0, 1).float().div_(127.5).sub_(1.)
        return video


class BatchSampler(Sampler):
    """
    A simple infinite batch sampler.
    """
    def __init__(self, dataset_size, batch_size, seed=2024):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed  # NOTE: ensure using different seeds for different ranks
        self.rng = np.random.default_rng(seed)
    
    def __iter__(self):
        while True:
            yield self.rng.choice(
                self.dataset_size,
                self.batch_size,
                replace=self.dataset_size < self.batch_size
            )


def collate_fn(batch):
    videos, texts, inner_ts = zip(*batch)
    return list(videos), list(texts), list(inner_ts)

def main(**kwargs):
    cfg.update(**kwargs)
    cfg.pmi_rank = int(os.environ['RANK'])
    cfg.pmi_world_size = int(os.environ['WORLD_SIZE'])
    cfg.gpus_per_machine = torch.cuda.device_count()
    cfg.world_size = cfg.pmi_world_size
    worker(cfg)


def worker(cfg):
    
    cfg.rank = cfg.pmi_rank
    gpu = cfg.rank % cfg.gpus_per_machine
    cfg.gpu = gpu
    print(f'rank: {cfg.pmi_rank} world_size: {cfg.pmi_world_size} gpus_per_machine: {cfg.gpus_per_machine}')

    # init distributed processes
    dist.init_process_group(
        backend='nccl',  # modern: 'cpu:gloo,cuda:nccl'
        rank=cfg.rank,
        world_size=cfg.world_size,
        timeout=datetime.timedelta(hours=5)
    )

    # logging
    reload(logging)
    if cfg.rank == 0:
        os.makedirs(cfg.log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(filename=cfg.log_file),
                logging.StreamHandler(stream=sys.stdout)
            ]
        )
        logging.info(cfg)
    else:
        logging.basicConfig(level=logging.ERROR)
    
    # disable warnings
    logging.getLogger('imageio_ffmpeg').setLevel(logging.ERROR)
    from torch.distributed.checkpoint._dedup_tensors import logger
    logger.setLevel(logging.ERROR)

    # [data] seeding
    cfg.seed += 1024 * cfg.rank
    rng = np.random.default_rng(cfg.seed)
    g = torch.Generator(device=gpu)
    g.manual_seed(cfg.seed)

    # [data] training
    logging.info('Initializing dataloader')
    dataset = VideoFolder(
        list_file=cfg.list_file,
        seq_len=cfg.seq_len,
        downsample=cfg.downsample,
        min_area=cfg.min_area,
        max_area=cfg.max_area,
        fps=cfg.fps,
        seed=cfg.seed,
        cfg=cfg,
    )
    sampler = BatchSampler(
        dataset_size=len(dataset),
        batch_size=cfg.batch_size,
        seed=cfg.seed
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor
    )
    rank_iter = iter(dataloader)

    # [model] init app and parse modules
    logging.info('Initializing T5, DAE, DiT, noise schedule, and diffusion process')

    ###
    resume_step = 0
    if cfg.resume_train:
        parent_dir = cfg.log_dir
        logging.info(f'loading the latest checkpoint from logging folder in cfg: {parent_dir}')

        checkpoint_path = cfg.checkpoint_path

        if os.path.exists(parent_dir):
            checkpoints_dir = os.path.join(parent_dir, "checkpoints")

            if os.path.exists(checkpoints_dir):
                # getting the latest step
                step_dirs = glob.glob(os.path.join(checkpoints_dir, "step_*"))
                max_step = -1
                for d in step_dirs:
                    try:
                        step = int(os.path.basename(d).split("_")[1])
                        if step > max_step:
                            max_step = step
                    except:
                        continue
                
                logging.info(f"finding the latest step: {max_step}")

                if max_step != -1:
                    # get the latest ckpt path
                    step_dir = os.path.join(checkpoints_dir, f"step_{max_step}")
                    new_ckpt = os.path.join(step_dir, "non_ema.pth")

                    if os.path.exists(new_ckpt):
                        checkpoint_path = new_ckpt
                        # ema_path = new_ckpt
                        logging.info(f"finding the latest ckpt: {checkpoint_path}, while ignoring the ckpt path within cfg file: {cfg.checkpoint_path}")
                    
                        resume_step = max_step
        
        checkpoint_list = [checkpoint_path, None]

        # boradcast the latest checkpoint path
        dist.broadcast_object_list(checkpoint_list, src=0)
        cfg.checkpoint_path, cfg.checkpoint_ema_path = checkpoint_list
    ###

    _app = WanxgenMulshot(
        device_id=gpu,
        fsdp_param_dtype=cfg.param_dtype,
        fsdp_reduce_dtype=cfg.reduce_dtype,
        fsdp_buffer_dtype=cfg.buffer_dtype,
        fsdp_sharding_strategy=cfg.sharding_strategy,
        cfg=cfg
    )


    t5, vae, dit, dit_ema, schedule, diffusion = (
        _app.t5, _app.vae, _app.dit, _app.dit_ema, _app.schedule, _app.diffusion
    )
    del _app


    # [optim] optimizer & acceleration
    optimizer = optim.AdamW(
        params=dit.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    scaler = ShardedGradScaler(enabled=True, process_group=None)
    micro_steps = cfg.num_steps * cfg.grad_accum

    # timing
    start = time.time()

    # training loop
    logging.info('Start the training loop')
    if resume_step != 0:
        logging.info(f'Skip data until step {resume_step}')
    for micro_step in range(1, micro_steps + 1):
        if micro_step <= resume_step:
            continue
        step = micro_step / cfg.grad_accum

        # read batch
        batch = next(rank_iter)
        batch = to_(batch, gpu, non_blocking=True)
        videos, texts, inner_t = batch

        # Sample a random timestep for each image without bias.
        u = compute_density_for_timestep_sampling(
            weighting_scheme='logit_normal',
            batch_size=len(videos),
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29, # Only effective when using the `'mode'` as the `weighting_scheme`
        )
        t = (u * schedule.config.num_train_timesteps).long()
        t = schedule.timesteps[t].to(device=videos[0].device)

        # preprocess
        with torch.no_grad():
            z = vae.encode(videos, inner_t=inner_t)
            context = []
            null = t5([''])
            for u in texts:
                if rng.random() < cfg.p_zero:
                    context.append(null * len(u))
                else:
                    context.append(t5(u))
        # forward
        with amp.autocast(dtype=cfg.param_dtype):
            loss = diffusion.loss(
                x0=z,
                t=t,
                model=dit,
                model_kwargs={'context': context, 'seq_len': cfg.seq_len, 'inner_t': inner_t},
                min_snr_gamma=cfg.min_snr_gamma,
                generator=g
            )
            loss = sum(loss) / len(loss)
        
        # backward
        scaler.scale(loss / cfg.grad_accum).backward()
        if micro_step % cfg.grad_accum == 0:
            # optimization step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # update ema
            if cfg.use_ema:
                with torch.no_grad():
                    ema = OrderedDict(dit_ema.named_parameters())
                    non_ema = OrderedDict(dit.named_parameters())
                    for k, v in ema.items():
                        v.copy_(non_ema[k].data.lerp(v, cfg.ema_decay))
                    del ema, non_ema

        # metrics
        dist.all_reduce(loss)
        loss /= cfg.world_size

        # logging
        if cfg.rank == 0 and (
            step == 1 or step % cfg.log_interval == 0 or step == cfg.num_steps
        ):
            batch_size = cfg.batch_size * cfg.world_size
            throughput = (86400 * micro_step * batch_size) / (time.time() - start)
            logging.info(
                f'Step: {int(step)}/{cfg.num_steps} '
                f'Loss: {loss.item():.2f} '
                f'lr: {optimizer.param_groups[0]["lr"]:.6f} '
                f'scale: {scaler.get_scale():.2f} '
                f'throughput: {round(throughput):d} samples/day'
            )
        
        # checkpoint
        if step == cfg.num_steps or step % cfg.val_interval == 0:
            _, _, free = shutil.disk_usage('/mnt/citysora-highspeed/home/huishi.wjh')
            if free / (1024**3) > 20:
                
                # saving path
                checkpoint_dir = osp.join(cfg.log_dir, f'checkpoints/step_{int(step)}')
                os.makedirs(checkpoint_dir, exist_ok=True)
                logging.info(f'Saving state dict to {checkpoint_dir}')

                # non-ema
                with FSDP.state_dict_type(
                    dit,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
                ):
                    non_ema = dit.state_dict()
                    if cfg.rank == 0:
                        torch.save(non_ema, osp.join(checkpoint_dir, 'non_ema.pth'))
                    del non_ema
                
                # ema
                if cfg.use_ema:
                    with FSDP.state_dict_type(
                        dit_ema,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
                    ):
                        ema = dit_ema.state_dict()
                        if cfg.rank == 0:
                            torch.save(ema, osp.join(checkpoint_dir, 'ema.pth'))
                        del ema
                
                # logging
                logging.info(f'Finished saving state dict. {int(free/(1024**3))} GB storage remaining.')
            else:
                logging.info(f'*** Skip saving state dict. {int(free/(1024**3))} GB storage remaining. ***')

    # barrier to ensure all ranks are completed
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()
    
    # completed!
    if cfg.rank == 0:
        logging.info('Congratulations! The training is completed!')


if __name__ == '__main__':
    from config_train import cfg
    main(cfg=cfg)
