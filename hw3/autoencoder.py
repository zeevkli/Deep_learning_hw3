import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod

class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=5, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu4 = nn.ReLU()

        self.modules = [self.conv1, self.bn1, self.relu1, self.conv2, self.bn2, self.relu2, self.conv3, self.bn3, self.relu3, self.conv4, self.bn4, self.relu4]
        # ========================
        self.cnn = nn.Sequential(*self.modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
        # ====== YOUR CODE: ======
        self.conv1 = nn.ConvTranspose2d(in_channels, 256, kernel_size=5, stride=2, padding=1, output_padding=0)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=1, output_padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=1, output_padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.ConvTranspose2d(64, out_channels, kernel_size=5, stride=2, padding=1, output_padding=1)
        self.modules = [self.conv1, self.bn1, self.relu1, self.conv2, self.bn2, self.relu2, self.conv3, self.bn3, self.relu3, self.conv4]
        # ========================
        self.cnn = nn.Sequential(*self.modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder that extracts features from an input.
        :param features_decoder: Instance of a decoder that reconstructs an input from its features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add more layers as needed for encode() and decode().
        # ====== YOUR CODE: ======
        self.enc_to_mu = nn.Linear(n_features, z_dim)
        self.enc_to_logvar = nn.Linear(n_features, z_dim)

        # Map z -> decoder features (same shape as encoder features output)
        self.z_to_dec = nn.Linear(z_dim, n_features)
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        # TODO:
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  1. Use the features extracted from the input to obtain mu and
        #     log_sigma2 (mean and log variance) of q(Z|x).
        #  2. Apply the reparametrization trick to obtain z.
        # ====== YOUR CODE: ======
        h = self.features_encoder(x)                 # (B, *features_shape)
        h_flat = h.view(h.shape[0], -1)              # (B, n_features)

        mu = self.enc_to_mu(h_flat)                  # (B, z_dim)
        log_sigma2 = self.enc_to_logvar(h_flat)      # (B, z_dim)

        # Reparameterization trick
        std = torch.exp(0.5 * log_sigma2)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, log_sigma2
        # ========================

    def decode(self, z):
        # TODO:
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer.
        #  2. Apply features decoder.
        # ====== YOUR CODE: ======
        h_flat = self.z_to_dec(z)                    # (B, n_features)
        h = h_flat.view(z.shape[0], *self.features_shape)  # (B, *features_shape)
        x_rec = self.features_decoder(h)
        # ========================
        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO:
            #  Sample from the model. Generate n latent space samples and
            #  return their reconstructions.
            #  Notes:
            #  - Remember that this means using the model for INFERENCE.
            #  - We'll ignore the sigma2 parameter here:
            #    Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #    the mean, i.e. psi(z).
            # ====== YOUR CODE: ======
            z = torch.randn(n, self.z_dim, device=device)

            # Decode to image space (mean of p(x|z))
            xr = self.decode(z)

            # Split batch into list of samples
            samples = list(xr)
            # ========================
        # Detach and move to CPU for display purposes.
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Returns (loss, data_loss, kldiv_loss), each a scalar averaged over batch.
    """
    x_sigma2 = torch.as_tensor(x_sigma2, device=x.device, dtype=x.dtype)

    # Data term: mean over C,H,W (per sample), then mean over batch
    # (No 1/2 factor in this homework's convention)
    per_sample_mse = (x - xr).pow(2).flatten(start_dim=1).mean(dim=1)  # (N,)
    data_loss = (per_sample_mse / x_sigma2).mean()                     # scalar

    # KL term: diagonal Gaussian KL to N(0,I)
    # (No 1/2 factor in this homework's convention)
    kld_per_sample = (torch.exp(z_log_sigma2) + z_mu.pow(2) - 1.0 - z_log_sigma2).sum(dim=1)  # (N,)
    kldiv_loss = kld_per_sample.mean()  # scalar

    loss = data_loss + kldiv_loss
    return loss, data_loss, kldiv_loss
