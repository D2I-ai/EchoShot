"""
ODE/SDE solvers for denoising diffusion models.
"""
import torch
import torchsde
import numbers
from tqdm.auto import trange

# from .schedules import ve, from_ve
from utils.utils import (
    randn_like,
    cache_video,
    rand_name,
    randn_like,
    to_,
    TensorList
)

__all__ = [
    'sample_euler',
    'sample_euler_ancestral',
    'sample_heun',
    'sample_dpm_2',
    'sample_dpm_2_ancestral',
    'sample_dpmpp_2s_ancestral',
    'sample_dpmpp_sde',
    'sample_dpmpp_2m',
    'sample_dpmpp_2m_sde',
    'SOLVERS'
]


#------------------------ conversion functions ------------------------#

def ve(sigma):
    """
    - vp:   xt = (1 - sigma ** 2) ** 0.5 * x0 + sigma * noise
    - ve:   xt = x0 + sigma * noise
    """
    assert isinstance(sigma, (numbers.Number, torch.Tensor))

    # get new sigma
    new_sigma = (sigma ** 2 / (1 - sigma ** 2)) ** 0.5
    
    # numeric issue
    if isinstance(sigma, numbers.Number):
        return new_sigma if sigma < 1. else float('inf')
    else:
        return torch.where(sigma < 1., new_sigma, float('inf'))


def from_ve(sigma):
    """
    - vp:   xt = (1 - sigma ** 2) ** 0.5 * x0 + sigma * noise
    - ve:   xt = x0 + sigma * noise
    """
    assert isinstance(sigma, (numbers.Number, torch.Tensor))

    # get new sigma
    new_sigma = (sigma ** 2 / (1 + sigma ** 2)) ** 0.5
    
    # numeric issue
    if isinstance(sigma, numbers.Number):
        return new_sigma if sigma < float('inf') else 1.
    else:
        return torch.where(sigma < float('inf'), new_sigma, 1.)

#-------------------- helper functions --------------------#

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """
    Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step.
    """
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (
        sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2
    ) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def get_scalings(sigma):
    return 1 / (1 + sigma ** 2) ** 0.5


#-------------------- variation exploding (VE) solvers --------------------#

@torch.no_grad()
def sample_euler(
    noise,
    model,
    sigmas,
    s_churn=0.,
    s_tmin=0.,
    s_tmax=float('inf'),
    s_noise=1.,
    seed=None,
    show_progress=True
):
    """
    Implements Algorithm 2 (Euler steps) from Karras et al. (2022).
    """
    g = None
    if seed is not None:
        g = torch.Generator(device=noise.device)
        g.manual_seed(seed)
    sigmas = ve(sigmas)
    x = noise * sigmas[0]
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        gamma = 0.
        if s_tmin <= sigmas[i] <= s_tmax and sigmas[i] < float('inf'):
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
        eps = randn_like(x, generator=g) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        # Euler method
        if sigmas[i] == float('inf'):
            denoised = model(noise, from_ve(sigma_hat))
            x = denoised + sigmas[i + 1] * (gamma + 1) * noise
        else:
            denoised = model(
                x * get_scalings(sigma_hat),
                from_ve(sigma_hat)
            )
            d = (x - denoised) / sigma_hat
            dt = sigmas[i + 1] - sigma_hat
            x = x + d * dt
    return x


@torch.no_grad()
def sample_euler_ancestral(
    noise,
    model,
    sigmas,
    eta=1.,
    s_noise=1.,
    seed=None,
    show_progress=True
):
    """
    Ancestral sampling with Euler method steps.
    """
    g = None
    if seed is not None:
        g = torch.Generator(device=noise.device)
        g.manual_seed(seed)
    sigmas = ve(sigmas)
    x = noise * sigmas[0]
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        # Euler method
        if sigmas[i] == float('inf'):
            denoised = model(noise, from_ve(sigmas[i]))
            x = denoised + sigmas[i + 1] * noise
        else:
            denoised = model(
                x * get_scalings(sigmas[i]),
                from_ve(sigmas[i])
            )
            d = (x - denoised) / sigmas[i]
            dt = sigma_down - sigmas[i]
            x = x + d * dt
            if sigmas[i + 1] > 0:
                x = x + randn_like(x, generator=g) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_heun(
    noise,
    model,
    sigmas,
    s_churn=0.,
    s_tmin=0.,
    s_tmax=float('inf'),
    s_noise=1.,
    seed=None,
    show_progress=True
):
    """
    Implements Algorithm 2 (Heun steps) from Karras et al. (2022).
    """
    g = None
    if seed is not None:
        g = torch.Generator(device=noise.device)
        g.manual_seed(seed)
    sigmas = ve(sigmas)
    x = noise * sigmas[0]
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        gamma = 0.
        if s_tmin <= sigmas[i] <= s_tmax and sigmas[i] < float('inf'):
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
        eps = randn_like(x, generator=g) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, from_ve(sigma_hat))
            x = denoised + sigmas[i + 1] * (gamma + 1) * noise
        else:
            denoised = model(
                x * get_scalings(sigma_hat),
                from_ve(sigma_hat)
            )
            d = (x - denoised) / sigma_hat
            dt = sigmas[i + 1] - sigma_hat
            if sigmas[i + 1] == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                denoised_2 = model(
                    x_2 * get_scalings(sigmas[i + 1]),
                    from_ve(sigmas[i + 1])
                )
                d_2 = (x_2 - denoised_2) / sigmas[i + 1]
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
    return x


@torch.no_grad()
def sample_dpm_2(
    noise,
    model,
    sigmas,
    s_churn=0.,
    s_tmin=0.,
    s_tmax=float('inf'),
    s_noise=1.,
    seed=None,
    show_progress=True
):
    """
    A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022).
    """
    g = None
    if seed is not None:
        g = torch.Generator(device=noise.device)
        g.manual_seed(seed)
    sigmas = ve(sigmas)
    x = noise * sigmas[0]
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        gamma = 0.
        if s_tmin <= sigmas[i] <= s_tmax and sigmas[i] < float('inf'):
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
        eps = randn_like(x, generator=g) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, from_ve(sigma_hat))
            x = denoised + sigmas[i + 1] * (gamma + 1) * noise
        else:
            denoised = model(
                x * get_scalings(sigma_hat),
                from_ve(sigma_hat)
            )
            d = (x - denoised) / sigma_hat
            if sigmas[i + 1] == 0:
                # Euler method
                dt = sigmas[i + 1] - sigma_hat
                x = x + d * dt
            else:
                # DPM-Solver-2
                sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
                dt_1 = sigma_mid - sigma_hat
                dt_2 = sigmas[i + 1] - sigma_hat
                x_2 = x + d * dt_1
                denoised_2 = model(
                    x_2 * get_scalings(sigma_mid),
                    from_ve(sigma_mid)
                )
                d_2 = (x_2 - denoised_2) / sigma_mid
                x = x + d_2 * dt_2
    return x


@torch.no_grad()
def sample_dpm_2_ancestral(
    noise,
    model,
    sigmas,
    eta=1.,
    s_noise=1.,
    seed=None,
    show_progress=True
):
    """
    Ancestral sampling with DPM-Solver second-order steps.
    """
    g = None
    if seed is not None:
        g = torch.Generator(device=noise.device)
        g.manual_seed(seed)
    sigmas = ve(sigmas)
    x = noise * sigmas[0]
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, from_ve(sigmas[i]))
            x = denoised + sigmas[i + 1] * noise
        else:
            denoised = model(
                x * get_scalings(sigmas[i]),
                from_ve(sigmas[i])
            )
            d = (x - denoised) / sigmas[i]
            if sigma_down == 0:
                # Euler method
                dt = sigma_down - sigmas[i]
                x = x + d * dt
            else:
                # DPM-Solver-2
                sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
                dt_1 = sigma_mid - sigmas[i]
                dt_2 = sigma_down - sigmas[i]
                x_2 = x + d * dt_1
                denoised_2 = model(
                    x_2 * get_scalings(sigma_mid),
                    from_ve(sigma_mid)
                )
                d_2 = (x_2 - denoised_2) / sigma_mid
                x = x + d_2 * dt_2
                x = x + randn_like(x, generator=g) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_dpmpp_2s_ancestral(
    noise,
    model,
    sigmas,
    eta=1.,
    s_noise=1.,
    seed=None,
    show_progress=True
):
    """
    Ancestral sampling with DPM-Solver++ (2S) second-order steps.
    """
    def t_to_sigma(t): return t.neg().exp()
    def sigma_to_t(sigma): return sigma.log().neg()

    g = None
    if seed is not None:
        g = torch.Generator(device=noise.device)
        g.manual_seed(seed)
    sigmas = ve(sigmas)
    x = noise * sigmas[0]
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, from_ve(sigmas[i]))
            x = denoised + sigma_down * noise
        else:
            denoised = model(
                x * get_scalings(sigmas[i]),
                from_ve(sigmas[i])
            )
            if sigma_down == 0:
                # Euler method
                d = (x - denoised) / sigmas[i]
                dt = sigma_down - sigmas[i]
                x = x + d * dt
            else:
                # DPM-Solver++(2S)
                t, t_next = sigma_to_t(sigmas[i]), sigma_to_t(sigma_down)
                r = 1 / 2
                h = t_next - t
                s = t + r * h
                x_2 = (t_to_sigma(s) / t_to_sigma(t)) * x - (-h * r).expm1() * denoised
                denoised_2 = model(
                    x_2 * get_scalings(t_to_sigma(s)),
                    from_ve(t_to_sigma(s))
                )
                x = (t_to_sigma(t_next) / t_to_sigma(t)) * x - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + randn_like(x, generator=g) * s_noise * sigma_up
    return x


class BatchedBrownianTree:
    """
    A wrapper around torchsde.BrownianTree that enables batches of entropy.
    """
    def __init__(self, x, t0, t1, seed=None, **kwargs):
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get('w0', torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2 ** 63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        self.trees = [torchsde.BrownianTree(
            t0, w0, t1, entropy=s, **kwargs
        ) for s in seed]
    
    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)
        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """
    A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """
    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x):
        self.transform = transform
        t0 = self.transform(torch.as_tensor(sigma_min)) - 1e-6  # avoid numeric issue
        t1 = self.transform(torch.as_tensor(sigma_max)) + 1e-6  # avoid numeric issue
        if isinstance(x, TensorList):
            self.tree = [BatchedBrownianTree(u.unsqueeze(0), t0, t1, seed) for u in x]
        else:
            self.tree = BatchedBrownianTree(x, t0, t1, seed)
    
    def __call__(self, sigma, sigma_next):
        t0 = self.transform(torch.as_tensor(sigma))
        t1 = self.transform(torch.as_tensor(sigma_next))
        if isinstance(self.tree, list):
            return TensorList([
                u(t0, t1).squeeze(0) / (t1 - t0).abs().sqrt() for u in self.tree
            ])
        else:
            return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


@torch.no_grad()
def sample_dpmpp_sde(
    noise,
    model,
    sigmas,
    eta=1.,
    s_noise=1.,
    r=1 / 2,
    seed=None,
    show_progress=True
):
    """
    DPM-Solver++ (stochastic).
    """
    def t_to_sigma(t): return t.neg().exp()
    def sigma_to_t(sigma): return sigma.log().neg()

    sigmas = ve(sigmas)
    x = noise * sigmas[0]
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas[sigmas < float('inf')].max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed)
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, from_ve(sigmas[i]))
            x = denoised + sigmas[i + 1] * noise
        else:
            denoised = model(
                x * get_scalings(sigmas[i]),
                from_ve(sigmas[i])
            )
            if sigmas[i + 1] == 0:
                # Euler method
                d = (x - denoised) / sigmas[i]
                dt = sigmas[i + 1] - sigmas[i]
                x = x + d * dt
            else:
                # DPM-Solver++
                t, t_next = sigma_to_t(sigmas[i]), sigma_to_t(sigmas[i + 1])
                h = t_next - t
                s = t + h * r
                fac = 1 / (2 * r)

                # Step 1
                sd, su = get_ancestral_step(t_to_sigma(t), t_to_sigma(s), eta)
                s_ = sigma_to_t(sd)
                x_2 = (t_to_sigma(s_) / t_to_sigma(t)) * x - (t - s_).expm1() * denoised
                x_2 = x_2 + noise_sampler(t_to_sigma(t), t_to_sigma(s)) * s_noise * su
                denoised_2 = model(
                    x_2 * get_scalings(t_to_sigma(s)),
                    from_ve(t_to_sigma(s))
                )

                # Step 2
                sd, su = get_ancestral_step(t_to_sigma(t), t_to_sigma(t_next), eta)
                t_next_ = sigma_to_t(sd)
                denoised_d = (1 - fac) * denoised + fac * denoised_2
                x = (t_to_sigma(t_next_) / t_to_sigma(t)) * x - \
                    (t - t_next_).expm1() * denoised_d
                x = x + noise_sampler(t_to_sigma(t), t_to_sigma(t_next)) * s_noise * su
    return x


@torch.no_grad()
def sample_dpmpp_2m(
    noise,
    model,
    sigmas,
    seed=None,
    show_progress=True
):
    """
    DPM-Solver++ (2M).
    """
    def t_to_sigma(t): return t.neg().exp()
    def sigma_to_t(sigma): return sigma.log().neg()

    sigmas = ve(sigmas)
    x = noise * sigmas[0]
    old_denoised = None
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, from_ve(sigmas[i]))
            x = denoised + sigmas[i + 1] * noise
        else:
            denoised = model(
                x * get_scalings(sigmas[i]),
                from_ve(sigmas[i])
            )
            t, t_next = sigma_to_t(sigmas[i]), sigma_to_t(sigmas[i + 1])
            h = t_next - t
            if old_denoised is None or \
                sigmas[i - 1] == float('inf') or sigmas[i + 1] == 0:
                x = (t_to_sigma(t_next) / t_to_sigma(t)) * x - (-h).expm1() * denoised
            else:
                h_last = t - sigma_to_t(sigmas[i - 1])
                r = h_last / h
                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                x = (t_to_sigma(t_next) / t_to_sigma(t)) * x - (-h).expm1() * denoised_d
            old_denoised = denoised
    return x


@torch.no_grad()
def sample_dpmpp_2m_sde(
    noise,
    model,
    sigmas,
    eta=1.,
    s_noise=1.,
    solver_type='midpoint',
    seed=None,
    show_progress=True
):
    """
    DPM-Solver++ (2M) SDE.
    """
    assert solver_type in {'heun', 'midpoint'}

    sigmas = ve(sigmas)
    x = noise * sigmas[0]
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas[sigmas < float('inf')].max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed)
    old_denoised = None
    h_last = None

    for i in trange(len(sigmas) - 1, disable=not show_progress):
        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, from_ve(sigmas[i]))
            x = denoised + sigmas[i + 1] * noise
        else:
            denoised = model(
                x * get_scalings(sigmas[i]),
                from_ve(sigmas[i])
            )
            if sigmas[i + 1] == 0:
                # Denoising step
                x = denoised
            else:
                # DPM-Solver++(2M) SDE
                t, s = -sigmas[i].log(), -sigmas[i + 1].log()
                h = s - t
                eta_h = eta * h

                x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + \
                    (-h - eta_h).expm1().neg() * denoised

                if old_denoised is not None:
                    r = h_last / h
                    if solver_type == 'heun':
                        x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * \
                            (1 / r) * (denoised - old_denoised)
                    elif solver_type == 'midpoint':
                        x = x + 0.5 * (-h - eta_h).expm1().neg() * \
                            (1 / r) * (denoised - old_denoised)

                x = x + noise_sampler(
                    sigmas[i],
                    sigmas[i + 1]
                ) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

            old_denoised = denoised
            h_last = h
    return x


#-------------------- dictionary of solvers --------------------#

SOLVERS = {
    'euler': sample_euler,
    'euler_ancestral': sample_euler_ancestral,
    'heun': sample_heun,
    'dpm2': sample_dpm_2,
    'dpm2_ancestral': sample_dpm_2_ancestral,
    'dpmpp_2s_ancestral': sample_dpmpp_2s_ancestral,
    'dpmpp_sde': sample_dpmpp_sde,
    'dpmpp_2m': sample_dpmpp_2m,
    'dpmpp_2m_sde': sample_dpmpp_2m_sde
}
