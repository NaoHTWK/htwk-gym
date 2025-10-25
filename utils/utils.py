import torch
import numpy as np


def apply_randomization(tensor, params, return_noise=False):
    if params == None:
        return tensor

    if params["distribution"] == "gaussian":
        mu, var = params["range"]
        noise = torch.randn_like(tensor) if isinstance(tensor, torch.Tensor) else np.random.randn()
        noise_val = mu + var * noise
    elif params["distribution"] == "uniform":
        lower, upper = params["range"]
        noise = torch.rand_like(tensor) if isinstance(tensor, torch.Tensor) else np.random.rand()
        noise_val = lower + (upper - lower) * noise
    else:
        raise ValueError(f"Invalid randomization distribution: {params['distribution']}")

    if params["operation"] == "additive":
        result = tensor + noise_val
    elif params["operation"] == "scaling":
        result = tensor * noise_val
    else:
        raise ValueError(f"Invalid randomization operation: {params['operation']}")

    if return_noise:
        return result, noise
    else:
        return result


def discount_values(rewards, dones, values, last_values, gamma, lam):
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(advantages[-1, :])
    for t in reversed(range(rewards.shape[0])):
        next_nonterminal = 1.0 - dones[t, :].float()
        if t == rewards.shape[0] - 1:
            next_values = last_values
        else:
            next_values = values[t + 1, :]
        delta = rewards[t, :] + gamma * next_nonterminal * next_values - values[t, :]
        advantages[t, :] = last_advantage = delta + gamma * lam * next_nonterminal * last_advantage
    return advantages


def surrogate_loss(old_actions_log_prob, actions_log_prob, advantages, e_clip=0.2):
    ratio = torch.exp(actions_log_prob - old_actions_log_prob)
    surrogate = -advantages * ratio
    surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - e_clip, 1.0 + e_clip)
    surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
    return surrogate_loss
