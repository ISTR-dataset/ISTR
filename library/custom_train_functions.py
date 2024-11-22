import torch
import argparse


def apply_snr_weight(loss, timesteps, noise_scheduler, gamma):

    alphas_cumprod = noise_scheduler.alphas_cumprod

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod

    all_snr = (alpha / sigma) ** 2

    snr = torch.stack([all_snr[t] for t in timesteps])

    gamma_over_snr = torch.div(torch.ones_like(snr) * gamma, snr)

    snr_weight = torch.minimum(gamma_over_snr, torch.ones_like(gamma_over_snr)).float()

    loss = loss * snr_weight
    return loss


def add_custom_train_arguments(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--min_snr_gamma",
        type=float,
        default=None,
        help="Gamma value used to reduce the weight of high-loss timesteps. The lower the value, the more significant the effect. "
    )
