import os
import os.path as osp
import torch
from easydict import EasyDict
import torch
from torch.distributed.fsdp import ShardingStrategy


H = 480
W = 832
T = 125

h = H // 16
w = W // 16
t = (T - 1) // 4 + 1
#------------------------ environment ------------------------#
cfg = EasyDict(__name__='Config: Wanx I2V')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['PYTHONHASHSEED'] = '2024'  # ensure consistent output from `hash()`
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

#---------------------------- data ---------------------------#
cfg.T, cfg.H, cfg.W = T, H, W
cfg.t, cfg.h, cfg.w = t, h, w

# dataset
cfg.list_file = './dataset/train.json'

cfg.h, cfg.w = h, w
cfg.seq_len =  t * h * w
cfg.min_area = round((H * W) * 1.0)
cfg.max_area = round((H * W) / 1.0)
cfg.fps = 16
cfg.seed = 2024

# sampler
cfg.resume_train = False
cfg.batch_size = 2
cfg.num_grad_checkpoints = 10   # or None

# dataloader
cfg.num_workers = 4
cfg.prefetch_factor = 2

#------------------------ parallelism ------------------------#

# FDSP sharding [SHARD_GRAD_OP | FULL_SHARD ｜ HYBRID_SHARD]
cfg.sharding_strategy = ShardingStrategy.NO_SHARD

# FSDP dtypes
cfg.param_dtype = torch.bfloat16
cfg.reduce_dtype = torch.float32
cfg.buffer_dtype = torch.float32

#------------------------ diffusion ------------------------#

# schedule
cfg.schedule = 'cosine'
cfg.num_timesteps = 1000
cfg.logsnr_scale_min = 2
cfg.logsnr_scale_max = 4

# diffusion
cfg.prediction_type = 'fm_scheduler'
cfg.min_snr_gamma = None
cfg.guide_rescale = None
cfg.guide_scale = 5.0
cfg.solver = 'dpmpp_2m_sde'
cfg.neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
cfg.pos_prompt = '32k，超高清，高度详细的，细节清晰可见，完美无缺失，较为缓慢的运动，连续的，平滑的，自然的'

# classifier-free guidance
cfg.p_zero = 0.1


#------------------------ model ------------------------#s
# t5
cfg.t5_model = 'umt5_xxl'
cfg.t5_dtype = torch.bfloat16
cfg.text_len = 512
cfg.t5_checkpoint = './models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth'
cfg.t5_tokenizer = './models/Wan2.1-T2V-1.3B/google/umt5-xxl/'

# vae
cfg.vae_checkpoint = './models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'
cfg.downsample = (4, 16, 16)

# transformer-1.3B
cfg.patch_size = (1, 2, 2)
cfg.dim = 1536
cfg.ffn_dim = 8960
cfg.freq_dim = 256
cfg.num_heads = 12
cfg.num_layers = 30
cfg.window_size = (-1, -1)
cfg.qk_norm = True
cfg.cross_attn_norm = True
cfg.eps = 1e-6
cfg.checkpoint_path = './models/Wan2.1-T2V-1.3B/wanx_t2v_1B.pth'
cfg.checkpoint_ema_path = './models/Wan2.1-T2V-1.3B/wanx_t2v_1B.pth'


############################################################
#------------------------ training ------------------------#
############################################################
# optimizer
cfg.num_steps = 1000000
cfg.lr = 8e-6
cfg.weight_decay = 0.001

# gradient accumulation
cfg.grad_accum = 1

# training
cfg.use_ema = False
cfg.ema_decay = 0.9999
cfg.ema_start_step = 0

# visualization
cfg.show_progress = True

# logging intervals
cfg.log_interval = 10
cfg.val_interval = 250

# logging path
cfg.log_dir = './runs/echoshot'
cfg.log_file = osp.join(cfg.log_dir, osp.basename(cfg.log_dir) + '.log')

# ----------------- inference ----------------------------

cfg.param_dtype = torch.bfloat16
cfg.num_train_timesteps = 1000


cfg.sample_fps = 16
cfg.sample_shift = 5.0
cfg.sample_guide_scale = 5.0
cfg.sample_steps = 50
cfg.sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
cfg.base_seed = 0
cfg.max_seq_len = t * h * w
cfg.target_shape = (16, t, H//8, W//8) # video size // 8

cfg.input_prompts = ['']
cfg.use_prompt_extend = False