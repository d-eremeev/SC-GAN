import torch
from torch.autograd import grad


def l2_loss_kmin(traj_real,
                 generator_,
                 image,
                 actor_state,
                 cfg,
                 kmin,
                 return_best_traj=False
                 ):
    """
    Apply current generator k times
    and select minimal L2 distance to real trajectory among generated trajectories as loss.
    """

    batch_size = image.shape[0]
    noise_dim = cfg['gan_params']['noise_dim']
    pred_len = cfg['model_params']['future_num_frames']

    l2_losses = []
    trajectories = []

    # generate k trajectories
    for i in range(kmin):
        noise = torch.normal(size=(batch_size, noise_dim), mean=0.0, std=1.0,
                             dtype=torch.float32, device=cfg['device'])

        traj_fake = generator_(image, actor_state, noise)
        trajectories.append(traj_fake)

        current_l2_loss = l2_loss(traj_fake, traj_real)  # (batch_size, )
        l2_losses.append(current_l2_loss)

    # stack all k_min losses to select the best one
    stacked_losses = torch.stack(l2_losses, dim=1)  # (batch_size, k_min)

    # indices of best l2 loss for each element in batch
    best_indices = stacked_losses.argmin(dim=-1)  # (batch_size,)

    # select best loss for each element in batch
    losses = torch.gather(stacked_losses, 1, best_indices[:, None])  # (batch_size, 1)

    if cfg['losses']['variety_l2_mode'] == 'average':
        # minimal average loss per trajectory point
        l2_loss_ = torch.sum(losses) / (batch_size*pred_len)
    elif cfg['losses']['variety_l2_mode'] == 'sum':
        # minimal summary loss per whole trajectory
        l2_loss_ = torch.sum(losses) / batch_size
    else:
        raise NotImplementedError

    # return corresponding best trajectories
    if return_best_traj:
        stacked_trajectories = torch.stack(trajectories, dim=1)  # (batch_size, k_min, target_size, 2)
        best_traj = stacked_trajectories[torch.arange(batch_size), best_indices]  # (batch_size, target_size, 2)
        return l2_loss_, best_traj
    else:
        return l2_loss_


def l2_loss(traj_fake, traj_real):
    """
    Returns summary losses for generated trajectories

    traj_fake: Tensor of shape # (batch_size, target_size, 2). Predicted trajectory.
    traj_real: Tensor of shape # (batch_size, target_size, 2). Ground truth predictions.
    """

    loss = (traj_real - traj_fake)**2  # (batch_size, target_size, 2)

    # batch of summary losses for each trajectory
    loss = loss.sum(dim=2).sum(dim=1)  # (batch_size,)
    return loss


def gradient_penalty(discrim, real_trajectory, fake_trajectory, in_image, in_actor_state, lambda_gp, device):
    """
    Calculates the gradient penalty loss for Wasserstein GAN with GP (https://arxiv.org/abs/1704.00028)

    in_image - scene context image
    in_actor_state - history of actor states
    """

    # Random weight term of shape (batch_size, target_size, 2) for interpolation between real and fake samples
    batch_size = real_trajectory.shape[0]
    epsilon = torch.rand(size=(batch_size, 1, 1), device=device)  # (batch_size, 1, 1)

    # Get random interpolation between real and fake samples
    interpolates = (epsilon * real_trajectory + ((1 - epsilon) * fake_trajectory)).requires_grad_(True)  # (batch_size, target_size, 2)
    d_interpolates = discrim(interpolates, in_image, in_actor_state)  # (batch_size, 1)

    fake = torch.ones(size=(batch_size, 1), requires_grad=False, device=device)

    # Get gradient of d_interpolates w.r.t. interpolates:
    # We use torch.autograd.grad and set grad_output==1.

    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # (batch_size, target_size, 2)

    gradients = gradients.view(gradients.size(0), -1)  # (batch_size, 2*target_size)
    gradient_penalty_ = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return lambda_gp * gradient_penalty_
