import os
import logging
import numpy as np
import pickle
from collections import defaultdict

from omegaconf import DictConfig
import hydra

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader


from layers import Generator, Discriminator
from losses import gradient_penalty, l2_loss_kmin

from utils import f_get_raster_image, weights_init, evaluate
from utils import TransformDataset, plot_traj_on_map, get_results_plot, print_statistics

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset
from l5kit.rasterization import build_rasterizer


@hydra.main(config_name='cfg')
def trainer(cfg: DictConfig) -> None:
    
    os.environ["L5KIT_DATA_FOLDER"] = cfg.l5kit_data_folder
    dm = LocalDataManager(None)

    logger = logging.getLogger(__name__)

    logger.info("Working directory : {}".format(os.getcwd()))

    logger.info("Load dataset...")

    train_cfg = cfg["train_data_loader"]
    valid_cfg = cfg["valid_data_loader"]

    # rasterizer
    rasterizer = build_rasterizer(cfg, dm)

    train_path = train_cfg["key"]
    train_zarr = ChunkedDataset(dm.require(train_path)).open(cached=False)

    logger.info(f"train_zarr {type(train_zarr)}")

    # loading custom mask (we mask static agents)
    logger.info(f"Loading mask in path {train_cfg['mask_path']}")
    custom_mask = np.load(train_cfg['mask_path'])
    logger.info(f"Length of training mask is: {custom_mask.sum()}")

    train_agent_dataset = AgentDataset(cfg, train_zarr, rasterizer, agents_mask=custom_mask)

    # transform dataset to the proper frame of reference
    train_dataset = TransformDataset(train_agent_dataset, cfg)

    if not train_cfg['subset'] == -1:
        train_dataset = Subset(train_dataset, np.arange(train_cfg['subset']))

    train_loader = DataLoader(train_dataset,
                              shuffle=train_cfg["shuffle"],
                              batch_size=train_cfg["batch_size"],
                              num_workers=train_cfg["num_workers"])

    logger.info(train_agent_dataset)

    # loading custom mask for validation dataset
    logger.info(f"Loading val mask in path {valid_cfg['mask_path']}")
    val_custom_mask = np.load(valid_cfg['mask_path'])
    logger.info(f"Length of validation mask is: {val_custom_mask.sum()}")

    valid_path = valid_cfg["key"]
    valid_zarr = ChunkedDataset(dm.require(valid_path)).open(cached=False)

    logger.info(f"valid_zarr {type(train_zarr)}")

    valid_agent_dataset = AgentDataset(cfg, valid_zarr, rasterizer, agents_mask=val_custom_mask)

    # transform validation dataset to the proper frame of reference
    valid_dataset = TransformDataset(valid_agent_dataset, cfg)

    if not valid_cfg['subset'] == -1:
        valid_dataset = Subset(valid_dataset, valid_cfg['subset'])

    valid_loader = DataLoader(
        valid_dataset,
        shuffle=valid_cfg["shuffle"],
        batch_size=valid_cfg["batch_size"],
        num_workers=valid_cfg["num_workers"]
    )

    logger.info(valid_agent_dataset)
    logger.info(f"# Full AgentDataset train: {len(train_agent_dataset)} #valid: {len(valid_agent_dataset)}")
    logger.info(f"# Actual AgentDataset train: {len(train_dataset)} #valid: {len(valid_dataset)}")

    n_epochs = cfg['train_params']['num_epochs']

    d_steps = cfg['train_params']['num_d_steps']
    g_steps = cfg['train_params']['num_g_steps']

    noise_dim = cfg['gan_params']['noise_dim']
    g_learning_rate = cfg['train_params']['g_learning_rate']
    d_learning_rate = cfg['train_params']['d_learning_rate']

    if cfg['gan_params']['gan_type'] == 'vanilla':
        cross_entropy = nn.BCELoss()

    generator = Generator(input_dim=cfg['gan_params']['input_dim'],
                          embedding_dim=cfg['gan_params']['embedding_dim'],
                          decoder_dim=cfg['gan_params']['decoder_dim'],
                          trajectory_dim=cfg['model_params']['future_num_frames'],
                          noise_dim=noise_dim,
                          backbone_type=cfg['gan_params']['backbone_type'],
                          embedding_type=cfg['gan_params']['embedding_type']
                          )

    generator.to(cfg['device'])
    generator.train()  # train mode
    
    W = cfg['raster_params']['raster_size'][0]
    discriminator = Discriminator(width=W,
                                  h_0=cfg['raster_params']['ego_center'][0]*W,
                                  w_0=cfg['raster_params']['ego_center'][1]*W,
                                  r=cfg['raster_params']['pixel_size'][0],
                                  sigma=cfg['gan_params']['sigma'],
                                  channels_num=cfg['model_params']['future_num_frames']+3,
                                  num_disc_feats=cfg['gan_params']['num_disc_feats'],
                                  input_dim=cfg['gan_params']['input_dim'],
                                  device=cfg['device'],
                                  gan_type=cfg['gan_params']['gan_type'],
                                  embedding_type=cfg['gan_params']['embedding_type'],
                                  lstm_embedding_dim=cfg['gan_params']['embedding_dim']
                                  )

    discriminator.to(cfg['device'])
    discriminator.apply(weights_init)
    discriminator.train()  # train mode

    if cfg['gan_params']['gan_type'] == 'wasserstein':
        optimizer_g = optim.RMSprop(generator.parameters(), lr=g_learning_rate)
        optimizer_d = optim.RMSprop(discriminator.parameters(), lr=d_learning_rate)
    elif cfg['gan_params']['gan_type'] == 'wasserstein_gp':
        betas = (0.0, 0.9)
        optimizer_g = optim.Adam(generator.parameters(), lr=g_learning_rate, betas=betas)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=d_learning_rate, betas=betas)
    else:
        optimizer_g = optim.Adam(generator.parameters(), lr=g_learning_rate)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=d_learning_rate)

    d_steps_left = d_steps
    g_steps_left = g_steps

    # variables for statistics
    d_full_loss = []
    g_full_loss = []
    gp_values = []
    l2_variety_values = []
    metric_vals = []

    # checkpoint dictionary
    checkpoint = {
        'G_losses': defaultdict(list),
        'D_losses': defaultdict(list),
        'counters': {
            't': None,
            'epoch': None,
        },
        'g_state': None,
        'g_optim_state': None,
        'd_state': None,
        'd_optim_state': None
    }

    id_batch = 0

    # total number of batches
    len_of_epoch = len(train_loader)

    for epoch in range(n_epochs):
        for batch in train_loader:
            batch = [tensor.to(cfg['device']) for tensor in batch]

            # Creates single raster image from sequence of images from l5kit's AgentDataset
            batch[0] = f_get_raster_image(cfg=cfg,
                                          images=batch[0],
                                          history_weight=cfg['model_params']['history_fading_weight'])

            (image, target_positions, target_availabilities,
             history_positions, history_yaws, centroid, world_to_image) = batch

            actor_state = (history_positions, history_yaws)

            batch_size = image.shape[0]

            # noise for generator
            noise = torch.normal(size=(batch_size, noise_dim),
                                 mean=0.0,
                                 std=1.0,
                                 dtype=torch.float32,
                                 device=cfg['device'])

            #######################################
            #       TRAIN DISCRIMINATOR
            #######################################

            # train discriminator (d_steps_left) times (using different batches)
            # train generator (g_steps_left) times (using different batches)

            if d_steps_left > 0:
                d_steps_left -= 1

                for pd in discriminator.parameters():  # reset requires_grad
                    pd.requires_grad = True  # they are set to False below in generator update

                # freeze generator while training discriminator
                for pg in generator.parameters():
                    pg.requires_grad = False

                discriminator.zero_grad()

                # generate fake trajectories (batch_size, target_size, 2) for current batch
                fake_trajectory = generator(image, actor_state, noise)

                # discriminator predictions (batch_size, 1) on real and fake trajectories
                d_real_pred = discriminator(target_positions, image, actor_state)
                d_g_pred = discriminator(fake_trajectory, image, actor_state)

                # loss
                if cfg['gan_params']['gan_type'] == 'vanilla':
                    # tensor with true/fake labels of size (batch_size, 1)
                    real_labels = torch.full((batch_size,), 1, dtype=torch.float, device=cfg['device'])
                    fake_labels = torch.full((batch_size,), 0, dtype=torch.float, device=cfg['device'])

                    real_loss = cross_entropy(d_real_pred, real_labels)
                    fake_loss = cross_entropy(d_g_pred, fake_labels)

                    total_loss = real_loss + fake_loss
                elif cfg['gan_params']['gan_type'] == 'wasserstein':  # D(fake) - D(real)
                    total_loss = torch.mean(d_g_pred) - torch.mean(d_real_pred)
                elif cfg['gan_params']['gan_type'] == 'wasserstein_gp':
                    gp_loss = gradient_penalty(discrim=discriminator,
                                               real_trajectory=target_positions,
                                               fake_trajectory=fake_trajectory,
                                               in_image=image,
                                               in_actor_state=actor_state,
                                               lambda_gp=cfg['losses']['lambda_gp'],
                                               device=cfg['device'])

                    total_loss = torch.mean(d_g_pred) - torch.mean(d_real_pred) + gp_loss
                else:
                    raise NotImplementedError

                # calculate gradients for this batch
                total_loss.backward()
                optimizer_d.step()

                # weight clipping for discriminator in pure Wasserstein GAN
                if cfg['gan_params']['gan_type'] == 'wasserstein':
                    c = cfg['losses']['weight_clip']
                    for p in discriminator.parameters():
                        p.data.clamp_(-c, c)

                d_full_loss.append(total_loss.item())

                if cfg['gan_params']['gan_type'] == 'wasserstein_gp':
                    gp_values.append(gp_loss.item())

            #######################################
            #         TRAIN GENERATOR
            #######################################

            elif g_steps_left > 0:  # we either train generator or discriminator on current batch
                g_steps_left -= 1

                for pd in discriminator.parameters():
                    pd.requires_grad = False  # avoid discriminator training

                # unfreeze generator
                for pg in generator.parameters():
                    pg.requires_grad = True

                generator.zero_grad()

                if cfg['losses']['use_variety_l2']:
                    l2_variety_loss, fake_trajectory = l2_loss_kmin(traj_real=target_positions,
                                                                    generator_=generator,
                                                                    image=image,
                                                                    actor_state=actor_state,
                                                                    cfg=cfg,
                                                                    kmin=cfg['losses']['k_min'],
                                                                    return_best_traj=True)
                else:
                    fake_trajectory = generator(image, actor_state, noise)

                d_g_pred = discriminator(fake_trajectory, image, actor_state)

                if cfg['gan_params']['gan_type'] == 'vanilla':
                    # while training generator we associate generated fake examples
                    # with real labels in order to measure generator quality
                    real_labels = torch.full((batch_size,), 1, dtype=torch.float, device=cfg['device'])
                    fake_loss = cross_entropy(d_g_pred, real_labels)
                elif cfg['gan_params']['gan_type'] in ['wasserstein', 'wasserstein_gp']:  # -D(fake)
                    fake_loss = -torch.mean(d_g_pred)
                else:
                    raise NotImplementedError

                if cfg['losses']['use_variety_l2']:
                    fake_loss += cfg['losses']['weight_variety_l2'] * l2_variety_loss

                    l2_variety_values.append(l2_variety_loss.item())

                fake_loss.backward()
                optimizer_g.step()

                g_full_loss.append(fake_loss.item())

            # renew d_steps_left, g_steps_left at the end of full discriminator-generator training cycle
            if d_steps_left == 0 and g_steps_left == 0:
                d_steps_left = d_steps
                g_steps_left = g_steps

            # print current model state on train dataset
            if (id_batch > 0) and (id_batch % cfg['train_params']['print_every_n_steps'] == 0):

                print_statistics(logger=logger,
                                 cfg=cfg,
                                 epoch=epoch,
                                 len_of_epoch=len_of_epoch,
                                 id_batch=id_batch,
                                 d_full_loss=d_full_loss,
                                 g_full_loss=g_full_loss,
                                 gp_values=gp_values,
                                 l2_variety_values=l2_variety_values,
                                 print_over_n_last=1000)

                # save rasterized image of 0th element of current batch
                plot_traj_on_map(cfg, 0, batch, generator, save_name=str(id_batch),
                                 save_directory=cfg['train_params']['image_sample_dir'])

            # Save checkpoint and evaluate the model
            if (id_batch > 0) and (id_batch % cfg['train_params']['checkpoint_every_n_steps'] == 0):
                checkpoint['counters']['t'] = id_batch
                checkpoint['counters']['epoch'] = epoch

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = evaluate(cfg, generator, valid_loader)
                metric_vals.append(metrics_val)

                with open('metric_vals_list.pkl', 'wb') as handle:
                    pickle.dump(metric_vals, handle, protocol=pickle.HIGHEST_PROTOCOL)

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))

                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                checkpoint_path = os.path.join(os.getcwd(), f"{cfg['model_name']}_{id_batch}.pt")
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

                results_df, metric_df = get_results_plot(d_full_loss,
                                                         g_full_loss,
                                                         metric_vals,
                                                         train_window_size=100,
                                                         val_window_size=10,
                                                         is_save=True)

                results_df.to_excel('results.xlsx', index=False)
                metric_df.to_excel('val_metrics.xlsx', index=False)

            id_batch = id_batch + 1


if __name__ == "__main__":
    trainer()
