import numpy as np
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision.models import mobilenet_v2
from efficientnet_pytorch import EfficientNet


class DifferentionalRasterizerLayer(nn.Module):
    """
    Differential trajectory rasterizer from the article https://arxiv.org/abs/2004.06247
    It provides an width x width image for each point of generated trajectory.
    Each image is in separate channel (trajectory grids).
    """

    def __init__(self,
                 width,
                 h_0,
                 w_0,
                 r,
                 sigma,
                 device):

        super().__init__()

        self.W = width
        self.h_0 = h_0
        self.w_0 = w_0
        self.r = r
        self.sigma = sigma
        self.pi_ = np.pi
        self.device = device
        self.m = MultivariateNormal(torch.zeros(2, device=self.device),
                                    torch.eye(2, device=self.device) * self.sigma**2)

    def forward(self, input):
        bs_ = input.shape[0]
        hs_ = input.shape[1]

        ones_ = torch.ones(size=(bs_, hs_, self.W, self.W), device=self.device)
        ranges_ = torch.range(0, self.W-1, device=self.device).reshape((1, 1, 1, -1))

        # element-wise product
        delta_i = ones_*ranges_
        delta_j = delta_i.permute((0, 1, 3, 2))

        delta_i = (delta_i - self.h_0)*self.r
        delta_j = (delta_j - self.w_0)*self.r

        # delta (bs_, hs_, W, W, 2)
        delta = torch.stack((delta_i, delta_j), dim=-1)
        Delta = delta - input.reshape((bs_, hs_, 1, 1, 2))

        G = torch.exp(self.m.log_prob(Delta))

        return G  # (bs_, hs_, W, W)


class Generator(nn.Module):
    """
    Generator uses scene rgb image and actor state data (coordinates + yaws).
    For image embedding we use pretrained MobileNet or EfficientNet as backbone.
    For actor state embedding we use shallow mlp layer or conv1d + lstm.
    """

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 decoder_dim,
                 trajectory_dim,
                 noise_dim,
                 backbone_type,
                 embedding_type):

        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.decoder_dim = decoder_dim
        self.trajectory_dim = trajectory_dim
        self.noise_dim = noise_dim
        self.embedding_type = embedding_type

        print('Backbone Type:', backbone_type)

        if backbone_type == 'mobilenet':
            self.backbone = mobilenet_v2(pretrained=True)
            self.extracted_features = list(self.backbone.classifier.children())[-1].out_features  # 1000
        elif 'efficientnet' in backbone_type:
            self.backbone = EfficientNet.from_pretrained(backbone_type)
            self.extracted_features = self.backbone._fc.out_features  # 1000
        else:
            raise NotImplementedError

        if self.embedding_type == 'mlp':
            self.state_encoder = nn.Linear(self.input_dim, self.embedding_dim)
        elif self.embedding_type == 'lstm':
            self.state_encoder = LSTMEncoder(embedding_dim=self.embedding_dim,
                                             h_dim=self.embedding_dim,
                                             dropout=0.0)
        else:
            raise NotImplementedError

        # we add noise_dim for latent variable (noise)
        self.decoder_1 = nn.Linear(self.extracted_features + self.embedding_dim + self.noise_dim,
                                   self.decoder_dim)
        # we predict flattened trajectory vector for both x and y
        self.decoder_2 = nn.Linear(self.decoder_dim, 2*self.trajectory_dim)

    def forward(self, image, actor_state, noise):
        # image = (batch_size, 3, W, W)
        # actor_state = [(batch_size, h_s, 2), (batch_size, h_s, 1)] - pair: ego-car coordinates + yaws history
        # noise = (batch_size, noise_dim)

        batch_size = image.shape[0]

        # flattened scene context for image
        scene_context = self.backbone(image)  # (batch_size, self.extracted_features)

        actor_state = torch.cat(actor_state, dim=-1)  # (batch_size, h_s, 3)

        # flatten input (x, y, angle) for shallow layer
        if self.embedding_type == 'mlp':
            actor_state = actor_state.reshape(batch_size, -1)  # (batch_size, 3*h_s)

        # actor_state embedding
        encoded_state = self.state_encoder(actor_state)  # (batch_size, self.embedding_dim)

        # concatenate all inputs
        concat = torch.cat([scene_context, encoded_state, noise], dim=-1)  # (batch_size, self.extracted_features +
                                                                           #  self.embedding_dim + self.noise_dim)

        trajectory_decoded = self.decoder_1(concat)  # (batch_size, decoder_dim)
        trajectory_decoded = self.decoder_2(trajectory_decoded)  # (batch_size, 2*trajectory_dim)

        # reshape flattened vector to get (x,y) pairs
        trajectory_decoded = trajectory_decoded.reshape(batch_size, self.trajectory_dim, 2)  # (batch_size, self.trajectory_dim, 2)

        return trajectory_decoded


class FusionDCGAN(nn.Module):
    """
    Classic DCGAN with fusion added for actor state.
    This class handles Vanilla GAN and Wasserstein GAN.
    """

    def __init__(self, nc, ndf,
                 input_dim,
                 gan_type,
                 embedding_type,
                 lstm_embedding_dim):

        super().__init__()

        self.nc = nc  # input channels number
        self.ndf = ndf  # ndf - base for inner channels number
        self.input_dim = input_dim
        self.embedding_type = embedding_type
        self.lstm_embedding_dim = lstm_embedding_dim

        # in case of lstm encoding we first lstm-encode and later project with nn.linear to proper dimension
        if self.embedding_type == 'lstm':
            self.initial_embedding = LSTMEncoder(embedding_dim=self.lstm_embedding_dim,
                                                 h_dim=self.lstm_embedding_dim,
                                                 dropout=0.0)
            self.input_dim = self.lstm_embedding_dim

        # embedding_dim = 3*18*18 since we want to reshape it to (3, 18, 18)
        self.state_encoder = nn.Linear(self.input_dim, 3*18*18)

        # Conv with kernel 1x1 for state fusion
        self.conv_state = nn.Conv2d(3, self.ndf * 8, kernel_size=1)

        self.conv_block1 = nn.Sequential(
            # input size is (nc) x 300 x 300
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_block2 = nn.Sequential(
            # input size is (ndf) x 150 x 150
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_block3 = nn.Sequential(
            # input size is (ndf*2) x 75 x 75
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_block4 = nn.Sequential(
            # input size is (ndf*4) x 37 x 37
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_block5 = nn.Sequential(
            # input size is(ndf*8) x 18 x 18
            nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_block6 = nn.Sequential(
            # input size is (ndf*16) x 9 x 9
            nn.Conv2d(self.ndf * 16, self.ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if gan_type == 'vanilla':
            self.conv_block7 = nn.Sequential(
                # input size is (ndf*32) x 4 x 4
                nn.Conv2d(self.ndf * 32, 1, 4, 1, 0, bias=False),
                # state size. 1 x 1 x 1
                nn.Sigmoid()
            )
        else:  # for wasserstein GAN
            self.conv_block7 = nn.Sequential(
                # input size is (ndf*32) x 4 x 4
                nn.Conv2d(self.ndf * 32, 1, 4, 1, 0, bias=False),
                # state size. 1 x 1 x 1
            )

    def forward(self, image, actor_state):
        # actor_state = [(batch_size, h_s, 2), (batch_size, h_s, 1)]

        batch_size = image.shape[0]

        actor_state = torch.cat(actor_state, dim=-1)  # (batch_size, h_s, 3)

        if self.embedding_type == 'mlp':
            actor_state = actor_state.reshape(batch_size, -1)  # (batch_size, 3*h_s)
        elif self.embedding_type == 'lstm':
            # add initial embedding in lstm-case
            actor_state = self.initial_embedding(actor_state)  # (batch_size, lstm_embeding_dim)
        else:
            raise NotImplementedError

        # actor_state fusion embedding
        encoded_state = self.state_encoder(actor_state)  # (batch_size, 3*18*18)
        encoded_state = encoded_state.reshape(batch_size, 3, 18, 18)  # (batch_size, 3, 18, 18)
        encoded_state = self.conv_state(encoded_state)  # (batch_size, ndf * 8, 18, 18)

        y = self.conv_block1(image)
        y = self.conv_block2(y)
        y = self.conv_block3(y)
        y = self.conv_block4(y)

        # fusion
        y = y + encoded_state

        y = self.conv_block5(y)
        y = self.conv_block6(y)
        y = self.conv_block7(y)

        return y


class FusionDCGAN_gp(nn.Module):
    """
    Classic DCGAN with fusion added for actor state.
    This class handles Wasserstein GAN with Gradient Penalty.
    There is no batchnorm in conv blocks.
    """

    def __init__(self, nc, ndf,
                 input_dim,
                 embedding_type,
                 lstm_embedding_dim):

        super().__init__()

        self.nc = nc  # input channels number
        self.ndf = ndf  # ndf - base for inner channels number
        self.input_dim = input_dim
        self.embedding_type = embedding_type
        self.lstm_embedding_dim = lstm_embedding_dim

        # in case of lstm encoding we first lstm-encode and later project with nn.linear to proper dimension
        if self.embedding_type == 'lstm':
            self.initial_embedding = LSTMEncoder(embedding_dim=self.lstm_embedding_dim,
                                                 h_dim=self.lstm_embedding_dim,
                                                 dropout=0.0)
            self.input_dim = self.lstm_embedding_dim

        # embedding_dim = 3*18*18 since we want to reshape it to (3, 18, 18)
        self.state_encoder = nn.Linear(self.input_dim, 3 * 18 * 18)  # torch.linear requires input dimensions

        # Conv with kernel 1x1 for state fusion
        self.conv_state = nn.Conv2d(3, self.ndf * 8, kernel_size=1)

        self.conv_block1 = nn.Sequential(
            # input size is (nc) x 300 x 300
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_block2 = nn.Sequential(
            # input size is (ndf) x 150 x 150
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_block3 = nn.Sequential(
            # input size is (ndf*2) x 75 x 75
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_block4 = nn.Sequential(
            # input size is (ndf*4) x 37 x 37
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_block5 = nn.Sequential(
            # input size is (ndf*8) x 18 x 18
            nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_block6 = nn.Sequential(
            # input size is (ndf*16) x 9 x 9
            nn.Conv2d(self.ndf * 16, self.ndf * 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_block7 = nn.Sequential(
            # input size is (ndf*32) x 4 x 4
            nn.Conv2d(self.ndf * 32, 1, 4, 1, 0, bias=False),
            # state size. 1 x 1 x 1
        )

    def forward(self, image, actor_state):
        # actor_state = [(batch_size, h_s, 2), (batch_size, h_s, 1)]

        batch_size = image.shape[0]

        actor_state = torch.cat(actor_state, dim=-1)  # (batch_size, h_s, 3)

        if self.embedding_type == 'mlp':
            actor_state = actor_state.reshape(batch_size, -1)  # (batch_size, 3*h_s)
        elif self.embedding_type == 'lstm':
            # add initial embedding in lstm-case
            actor_state = self.initial_embedding(actor_state)  # (batch_size, lstm_embeding_dim)
        else:
            raise NotImplementedError

        # actor_state fusion embedding
        encoded_state = self.state_encoder(actor_state)  # (batch_size, 3*18*18)
        encoded_state = encoded_state.reshape(batch_size, 3, 18, 18)  # (batch_size, 3, 18, 18)
        encoded_state = self.conv_state(encoded_state)  # (batch_size, ndf * 8, 18, 18)

        y = self.conv_block1(image)
        y = self.conv_block2(y)
        y = self.conv_block3(y)
        y = self.conv_block4(y)

        # fusion
        y = y + encoded_state

        y = self.conv_block5(y)
        y = self.conv_block6(y)
        y = self.conv_block7(y)

        return y


class Discriminator(nn.Module):
    """
    Discriminator that uses differentiable  rasterization from article https://arxiv.org/abs/2004.06247
    for generated trajectory along with scene context image
    and history of actor states, that we add via so-called Fusion, as  proposed in https://arxiv.org/abs/1906.08469 .

    Three types of GAN architectures supported: Vanilla DCGAN, Wasserstein GAN and Wasserstein GAN with Gradient Penalty.
    """

    def __init__(self,
                 width,
                 h_0,
                 w_0,
                 r,
                 sigma,
                 channels_num,
                 num_disc_feats,
                 input_dim,
                 device,
                 gan_type,
                 embedding_type,
                 lstm_embedding_dim):
        super().__init__()

        self.width = width
        self.h_0 = h_0
        self.w_0 = w_0
        self.r = r
        self.sigma = sigma
        self.channels_num = channels_num
        self.num_disc_feats = num_disc_feats
        self.input_dim = input_dim
        self.device = device
        self.gan_type = gan_type
        self.embedding_type = embedding_type
        self.lstm_embedding_dim = lstm_embedding_dim

        self.diff_rasterizer = DifferentionalRasterizerLayer(self.width,
                                                             self.h_0,
                                                             self.w_0,
                                                             self.r,
                                                             self.sigma,
                                                             self.device)

        if gan_type == 'wasserstein_gp':
            self.fusion_dcgan = FusionDCGAN_gp(nc=self.channels_num,
                                               ndf=self.num_disc_feats,
                                               input_dim=self.input_dim,
                                               embedding_type=self.embedding_type,
                                               lstm_embedding_dim=self.lstm_embedding_dim
                                               )
        else:  # vanilla/wasserstein
            self.fusion_dcgan = FusionDCGAN(nc=self.channels_num,
                                            ndf=self.num_disc_feats,
                                            input_dim=self.input_dim,
                                            gan_type=self.gan_type,
                                            embedding_type=self.embedding_type,
                                            lstm_embedding_dim=self.lstm_embedding_dim
                                            )

    def forward(self,
                trajectory,
                image,
                actor_state):
        # trajectory: (batch_size, target_size, 2) - predicted or ground truth trajectory
        # image: (batch_size, 3, W, W) - rasterized scene context image
        # actor_state = [(batch_size, h_s, 2), (batch_size, h_s, 1)] - coordinates + yaws

        batch_size = image.shape[0]

        # generate N = target_size images (grids)
        # there is an image for each point in predicted/ground truth trajectory
        trajectory_grids = self.diff_rasterizer(trajectory)  # (batch_size, target_size, W, W)

        # add scene context image
        trajectory_grids = torch.cat((trajectory_grids, image), dim=1)  # (batch_size, target_size+3, W, W)
        # add history of actor states via Fusion and process through DCGAN
        out = self.fusion_dcgan(trajectory_grids, actor_state)  # (batch_size, 1, 1, 1)
        out = out.reshape(batch_size, -1)

        return out


class LSTMEncoder(nn.Module):
    """
    Conv1D + LSTM layers to encode a sequence of ego-car's coordinates and yaws.
    """

    def __init__(self,
                 embedding_dim=128,
                 h_dim=128,
                 dropout=0.0):

        super().__init__()

        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = 1

        # (bs, history_size, 2) --> (bs, history_size, 128)
        self.spatial_embedding = Conv1DEmbedder(embedding_dim=self.embedding_dim)

        self.encoder = nn.LSTM(self.embedding_dim,
                               self.h_dim,
                               self.num_layers,
                               dropout=dropout)

    def forward(self, actor_state):
        """
        Inputs: concatenated tuple of tensors
        - history_positions: (batch, history_size, 2)
        - history_yaws: (batch, history_size, 1)

        Output:
        - final_h: (batch, self.h_dim)
        """

        # encode trajectory
        history_data_embedding = self.spatial_embedding(actor_state)

        output, state = self.encoder(history_data_embedding.permute(1, 0, 2))  # lstm input is (seq_len, batch, input_size)

        final_h = state[0]

        return final_h.permute(1, 0, 2).squeeze(1)


class Conv1DEmbedder(nn.Module):
    def __init__(self, embedding_dim=128):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.conv1d = nn.Conv1d(3, self.embedding_dim, 3, padding=1)

    def forward(self, history_data):
        history_data = history_data.permute(0, 2, 1)  # (bs, 3, history_size)
        history_data = self.conv1d(history_data)  # (bs, embedding_dim, history_size)
        return history_data.permute(0, 2, 1)  # (bs, history_size, embedding_dim)
