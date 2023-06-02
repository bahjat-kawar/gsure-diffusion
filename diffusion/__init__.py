
from . import gaussian_diffusion as gd
from .gaussian_diffusion import get_beta_schedule
from .respace import SpacedDiffusion, space_timesteps


def create_diffusion(
    timestep_respacing,
    beta_sched_params,
    mean_type=gd.ModelMeanType.EPSILON,
    diffusion_steps=1000,
):
    betas = get_beta_schedule(**beta_sched_params)
    loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=mean_type,
        model_var_type=gd.ModelVarType.FIXED_SMALL, loss_type=loss_type
    )

