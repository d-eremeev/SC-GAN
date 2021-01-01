import os
import gc

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns

import torch
from torch import nn
from torch.utils.data.dataset import Dataset

from l5kit.geometry import transform_points

from losses import l2_loss_kmin


def f_get_raster_image(cfg,
                       images,
                       history_weight=0.9):
    """
    Creates single raster image from sequence of images from l5kit's AgentDataset
        Args:
            cfg {dict}: Dictionary config.
            images: (batch_size, 2*(history_num_frames+1)+3, H, H) - sequences of images after applying l5kit's rasterizer:
                    There is (history_num_frames+1) ego-car images, (history_num_frames+1) agent-car's images + 3 scene RGB images
            history_weight {float}: Amount of history fading (for rendering).

        Returns:
            RGB image of the scene and agents.
            Red color stays for EGO car.
            Yellow color stays for Agent's cars.
    """

    batch_size = images.shape[0]
    image_size = images.shape[-1]

    # get number of history steps
    hnf = cfg['model_params']['history_num_frames']

    # define ego-car's indices range in channels (images):
    # ind (0, hnf) correspond to all agents except ego car,
    # from hnf+1 to 2*hnf+1 correspond to ego car,
    # last 3 indices correspond to rgb scene
    ego_index = range(hnf+1, 2*hnf+2)

    # iterate through ego-car's frames and sum them according to history_weight (history fading) in single channel.
    ego_path_image = torch.zeros(size=(batch_size, image_size, image_size), device=cfg['device'])
    for im_id in reversed(ego_index):
        ego_path_image = (images[:, im_id, :, :] + ego_path_image * history_weight).clamp(0, 1)

    # define agent's range
    agents_index = range(0, hnf+1)

    # iterate through agent-car's frames and sum them according to history_weight in single channel
    agents_path_image = torch.zeros(size=(batch_size, image_size, image_size), device=cfg['device'])
    for im_id in reversed(agents_index):
        agents_path_image = (images[:, im_id, :, :] + agents_path_image*history_weight).clamp(0, 1)

    #  RGB path for ego (red (255, 0, 0)); channels last
    ego_path_image_rgb = torch.zeros((ego_path_image.shape[0],
                                      ego_path_image.shape[1],
                                      ego_path_image.shape[2],
                                      3), device=cfg['device'])

    ego_path_image_rgb[:, :, :, 0] = ego_path_image

    # RGB paths for agents (yellow (255, 255, 0)); channels last
    agents_path_image_rgb = torch.zeros((agents_path_image.shape[0],
                                         agents_path_image.shape[1],
                                         agents_path_image.shape[2],
                                         3), device=cfg['device'])
    # yellow
    agents_path_image_rgb[:, :, :, 0] = agents_path_image
    agents_path_image_rgb[:, :, :, 1] = agents_path_image

    # generate full RGB image with all cars (ego + agents)
    all_vehicles_image = ego_path_image_rgb + agents_path_image_rgb  # (batch_size, 3, H, H)

    # get RGB image for scene from rasterizer (3 last images); channels last
    scene_image_rgb = images[:, 2*hnf+2:, :, :].permute(0, 2, 3, 1)

    # Add mask to positions of cars (to merge all layers):
    # We need to take into account that the off-road is white, i.e. 1 in all channels
    # So, we have to prevent disappearing of off-road cars after clipping when we add images together.
    # For ex. (1, 1, 1) + (1, 0, 0) = (2, 1, 1) --> (1, 1, 1) = off-road car disappears.
    # In order to solve this, we cut the scene at the car's area.

    scene_image_rgb[(all_vehicles_image > 0).any(dim=-1)] = 0.0

    # generate final raster map
    full_raster_image = (all_vehicles_image + scene_image_rgb).clamp(0, 1)

    # channels as a second dimension
    full_raster_image = full_raster_image.permute(0, 3, 1, 2)

    return full_raster_image  # (batch_size, 3, W, W)


# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname != 'Conv1DEmbedder':
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def evaluate(cfg, predictor, data_loader):
    """
    Evaluates model on the given dataset.
    Returns a dictionary with metrics.
    """

    predictor.eval()

    metrics = {}
    loss_l2_bo3_list = []
    loss_l2_bo20_list = []

    with torch.no_grad():

        for batch in data_loader:

            batch = [tensor.to(cfg['device']) for tensor in batch]
            batch[0] = f_get_raster_image(cfg=cfg,
                                          images=batch[0],
                                          history_weight=cfg['model_params']['history_fading_weight'])

            (image, target_positions, target_availabilities,
             history_positions, history_yaws, centroid, world_to_image) = batch

            actor_state = (history_positions, history_yaws)

            # best of k average l2 loss (per trajectory/point)
            loss_l2_bo3 = l2_loss_kmin(traj_real=target_positions,
                                       generator_=predictor,
                                       image=image,
                                       actor_state=actor_state,
                                       cfg=cfg,
                                       kmin=3)  # (1,)

            loss_l2_bo20 = l2_loss_kmin(traj_real=target_positions,
                                        generator_=predictor,
                                        image=image,
                                        actor_state=actor_state,
                                        cfg=cfg,
                                        kmin=20)  # (1,)

            loss_l2_bo3_list.append(loss_l2_bo3)
            loss_l2_bo20_list.append(loss_l2_bo20)

    loss_l2_bo3 = torch.stack(loss_l2_bo3_list, dim=0)  # (num_batches, )
    loss_l2_bo20 = torch.stack(loss_l2_bo20_list, dim=0)

    metrics['l2_best_of_3_loss'] = loss_l2_bo3.mean().item()
    metrics['l2_best_of_20_loss'] = loss_l2_bo20.mean().item()

    predictor.train()
    return metrics


def plot_traj_on_map(cfg,
                     batch_id,  # id of image in the batch to be plotted
                     batch,
                     predictor,
                     figsize=(12, 12),
                     dpi=200,
                     is_show=False,
                     save_name=None,
                     save_directory=r'D:\kaggle_data\Lyft'
                     ):
    """
    Generate trajectories with current generator.
    Visualize history, target and predictions.
    """

    predictor.eval()

    (image, target_positions, target_availabilities,
     history_positions, history_yaws, centroid, world_to_image) = batch

    actor_state = (history_positions, history_yaws)

    batch_size = image.shape[0]
    noise_dim = cfg['gan_params']['noise_dim']

    # generate batch of predictions
    noise = torch.normal(size=(batch_size, noise_dim), mean=0.0, std=1.0, dtype=torch.float32, device=cfg['device'])
    gen_batch = predictor(image, actor_state, noise)

    xy = gen_batch[batch_id].cpu().detach().numpy()  # trajectory

    # We want trajectories in image frame of reference, but all trajectories are given in agents FoR
    # Trajectories are already properly rotated
    W = cfg['raster_params']['raster_size'][0]
    # agent's position (in pixels)
    agent_position_image = np.array([cfg['raster_params']['ego_center'][0] * W,
                                     cfg['raster_params']['ego_center'][1] * W])

    # transform to pixels by division on r and then translate
    r = cfg['raster_params']['pixel_size'][0]
    target_positions_pixels_pred = agent_position_image + xy/r  # (target_size, 2)
    target_positions_pixels_history = agent_position_image + history_positions[batch_id].cpu().numpy()/r
    target_positions_pixels_future = agent_position_image + target_positions[batch_id].cpu().numpy()/r

    #############
    #  Plotting
    #############

    plt.figure(figsize=figsize, dpi=dpi)

    # get rasterized image
    plt.imshow(image[batch_id].permute(1, 2, 0).cpu().numpy())  # (W, W, 3)

    plt.plot(target_positions_pixels_pred[:, 0],
             target_positions_pixels_pred[:, 1],
             linewidth=3,
             c='#46BCDE')

    plt.plot(target_positions_pixels_future[:, 0],
             target_positions_pixels_future[:, 1],
             linewidth=3,
             c='#D6D8DE',
             linestyle=':')

    plt.scatter(target_positions_pixels_history[:, 0],
                target_positions_pixels_history[:, 1],
                s=3,
                c='#52D273')

    plt.tight_layout()
    plt.axis('off')

    plt.legend(['prediction',
                'ground truth',
                'history'])

    if is_show:
        plt.show()
    else:
        plt.savefig(os.path.join(save_directory, f"{save_name}.jpg"))
        plt.close()
        # to prevent memory leak
        plt.ioff()
        plt.clf()
        # close all figure windows.
        plt.close('all')
        gc.collect()

    predictor.train()


def get_results_plot(d_full_loss,
                     g_full_loss,
                     metric_vals_list,
                     train_window_size=20,  # window size for generator/discriminator losses
                     val_window_size=10,  # window size for metrics on validation
                     is_save=False):

    # plotting train metrics (discriminator and generator)
    results_df = pd.DataFrame({'batch_number': list(range(len(d_full_loss))),
                               'd_loss': d_full_loss,
                               'g_loss': np.pad(np.array(g_full_loss), (0, len(d_full_loss) - len(g_full_loss)),
                                                'constant', constant_values=(0,))
                               })
    results_df['d_loss_ema'] = results_df.d_loss.rolling(window=train_window_size).mean()
    results_df['g_loss_ema'] = results_df.g_loss.rolling(window=train_window_size).mean()

    ###########################
    # plot discriminator loss
    ###########################

    plt.figure(figsize=(12, 9))
    sns.lineplot(x='batch_number', y='d_loss', data=results_df, color='cornflowerblue')
    sns.lineplot(x='batch_number', y='d_loss_ema', data=results_df, color='mediumblue')
    plt.title('Discriminator loss')

    if is_save:
        plt.savefig('d_loss_training.jpg')
        plt.close()
    else:
        plt.show()

    ###########################
    # plot generator loss
    ###########################

    plt.figure(figsize=(12, 9))
    sns.lineplot(x='batch_number', y='g_loss', data=results_df.loc[results_df['g_loss'] != 0], color='bisque')
    sns.lineplot(x='batch_number', y='g_loss_ema', data=results_df.loc[results_df['g_loss'] != 0], color='orange')
    plt.title('Generator loss')

    if is_save:
        plt.savefig('g_loss_training.jpg')
        plt.close()
    else:
        plt.show()

    ###########################
    # plot validation metrics
    ###########################

    metric_df = pd.DataFrame(metric_vals_list)
    metric_names = metric_df.columns

    for metric_col in metric_names:
        metric_df[f'{metric_col}_ema'] = metric_df[metric_col].rolling(window=val_window_size).mean()

    metric_df = metric_df.reset_index()
    val_iteration = 'Val Iteration'
    metric_df = metric_df.rename(columns={'index': val_iteration})

    for metric_col in metric_names:
        plt.figure(figsize=(12, 9))
        sns.lineplot(x=val_iteration, y=metric_col, data=metric_df, color='cornflowerblue')
        sns.lineplot(x=val_iteration, y=f'{metric_col}_ema', data=metric_df, color='mediumblue')
        plt.title(f'Validation plot for {metric_col}')

        if is_save:
            plt.savefig(f'val_{metric_col}_metric.jpg')
            plt.close()
        else:
            plt.show()

    return results_df, metric_df


def print_statistics(logger,
                     cfg,
                     epoch,
                     len_of_epoch,
                     id_batch,
                     d_full_loss,
                     g_full_loss,
                     gp_values,
                     l2_variety_values,
                     print_over_n_last=1000):

    epoch_message = 'epoch = {} / {}, t = {} / {}'.format(epoch + 1,
                                                          cfg['train_params']['num_epochs'],
                                                          id_batch + 1,
                                                          len_of_epoch)

    d_loss_message = '  [D] {}: {:.3f}'.format('D batch Loss', round(d_full_loss[-1], 3))

    d_loss_avg_last_message = '  [D] {}: {:.3f}'.format(f'D mean Loss over {print_over_n_last} last:',
                                                        np.array(d_full_loss[-print_over_n_last:]).mean().round(3))

    if cfg['gan_params']['gan_type'] == 'wasserstein_gp':
        d_gp_loss_message = '  [D] {}: {:.3f}'.format('D batch GP Loss', round(gp_values[-1], 3))
        d_gp_loss_avg_last_message = '  [D] {}: {:.3f}'.format(f'D mean GP Loss over {print_over_n_last} last:',
                                                               np.array(gp_values[-print_over_n_last:]).mean().round(3))

    g_loss_message = '  [G] {}: {:.3f}'.format('G batch Loss', round(g_full_loss[-1], 3))
    g_loss_avg_last_message = '  [G] {}: {:.3f}'.format(f'G mean Loss over {print_over_n_last} last:',
                                                        np.array(g_full_loss[-print_over_n_last:]).mean().round(3))

    if cfg['losses']['use_variety_l2']:
        g_l2_loss_message = '  [G] {}: {:.3f}'.format('G batch L2 variety Loss',
                                                      round(l2_variety_values[-1], 3))
        g_l2_loss_avg_last_message = '  [G] {}: {:.3f}'.format(f'G mean L2 variety Loss over {print_over_n_last} last:',
                                                               np.array(l2_variety_values[-print_over_n_last:]).mean().round(3))

    logger.info(epoch_message)

    logger.info(d_loss_message)
    logger.info(d_loss_avg_last_message)

    if cfg['gan_params']['gan_type'] == 'wasserstein_gp':
        logger.info(d_gp_loss_message)
        logger.info(d_gp_loss_avg_last_message)

    logger.info(g_loss_message)
    logger.info(g_loss_avg_last_message)

    if cfg['losses']['use_variety_l2']:
        logger.info(g_l2_loss_message)
        logger.info(g_l2_loss_avg_last_message)


class TransformDataset(Dataset):
    def __init__(self, dataset, cfg):
        self.cfg = cfg
        self.dataset = dataset
        self.W = self.cfg['raster_params']['raster_size'][0]

    def __getitem__(self, index):
        batch = self.dataset[index]
        return self.transform(batch)

    def __len__(self):
        return len(self.dataset)

    # Here batch is just 1 element
    def transform(self, batch):

        # (2,) agent coordinates in image frame of reference
        agent_position_image = np.array([self.cfg['raster_params']['ego_center'][0]*self.W,
                                         self.cfg['raster_params']['ego_center'][1]*self.W])

        # (1,) meters per pixel
        r = self.cfg['raster_params']['pixel_size'][0]

        # initially positions are given in agent frame of reference but with wrong rotation
        # transform into image frame of reference using rotation and affine transformation
        # after that tranform into agent frame of reference back

        batch["target_positions"] = (transform_points(batch["target_positions"] + batch["centroid"],
                                                      batch["world_to_image"]) - agent_position_image)*r

        batch['history_positions'] = (transform_points(batch['history_positions'] + batch["centroid"],
                                                       batch["world_to_image"]) - agent_position_image)*r

        return (batch["image"].astype(np.float32),
                batch["target_positions"].astype(np.float32),
                batch["target_availabilities"].astype(np.float32),
                batch['history_positions'].astype(np.float32),
                batch['history_yaws'].astype(np.float32),
                batch["centroid"].astype(np.float32),
                batch["world_to_image"].astype(np.float32)
                )
