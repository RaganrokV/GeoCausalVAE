import torch
from torch import nn
from torch.nn import functional as F


def kl_normal(qm, qv, pm, pv):
    """
    KL(q||p)
    """
    element_wise = (
        0.5
        * (
            torch.log(pv)
            - torch.log(qv)
            + qv / pv
            + (qm - pm).pow(2) / pv
            - 1
        )
    )

    kl = element_wise.sum(-1)

    return kl


class VanillaVAE(nn.Module):

    def __init__(
            self,
            input_dim,
            z_dim=13
    ):
        super().__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim

        # ===== Encoder =====

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ELU(),

            nn.Linear(32, 16),
            nn.ELU(),

            nn.Linear(16, 2 * z_dim)
        )

        # ===== Decoder =====

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 16),
            nn.ELU(),

            nn.Linear(16, 32),
            nn.ELU(),

            nn.Linear(32, input_dim)
        )

    def encode(self, x):

        h = self.encoder(x)

        mu, var = h.chunk(2, dim=-1)

        var = F.softplus(var) + 1e-8

        return mu, var

    def reparameterize(self, mu, var):

        std = torch.sqrt(var)

        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):

        return self.decoder(z)

    def forward(self, x):

        mu, var = self.encode(x)

        z = self.reparameterize(mu, var)

        recon_x = self.decode(z)

        return recon_x, mu, var

    def negative_elbo_bound(self, x):

        recon_x, mu, var = self.forward(x)

        # reconstruction

        rec = (
            F.mse_loss(
                recon_x,
                x,
                reduction="sum"
            )
            / x.size(0)
        )

        # KL

        p_m = torch.zeros_like(mu)

        p_v = torch.ones_like(var)

        kl = kl_normal(
            mu,
            var,
            p_m,
            p_v
        ).mean()

        nelbo = rec + kl

        return nelbo, kl, rec, recon_x

    def loss(self, x):

        nelbo, kl, rec, recon_x = self.negative_elbo_bound(x)

        summaries = {
            "loss/total": nelbo.item(),
            "loss/kl": kl.item(),
            "loss/reconstruction": rec.item(),
            "metrics/mse": rec.item()
        }

        return nelbo, summaries
