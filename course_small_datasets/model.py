import torch
import torch.nn as nn



class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, act=None, use_shortcut=True):
        super(ResidualBlock, self).__init__()
        self.use_shortcut = use_shortcut
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(num_features=out_features)
        self.act = act if act is not None else nn.ReLU()

        if use_shortcut:
            if in_features != out_features:
                self.shortcut = nn.Linear(in_features, out_features)
            else:
                self.shortcut = nn.Identity()
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.act(self.bn(self.linear(x)))

        if self.use_shortcut:
            identity = self.shortcut(x)
            out = out + identity
        
        return out

pair = lambda l: zip(l[:-1], l[1:])
        
class Encoder(nn.Module):
    def __init__(self, D_in, hidden_dims, latent_dim, use_shortcut=False):
        super(Encoder, self).__init__()
                
        self.net = nn.ModuleList()
        for i, j in pair([D_in] + hidden_dims):
            self.net.append(ResidualBlock(i, j, use_shortcut=use_shortcut))
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, x):
        for block in self.net:
            x = block(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, D_out, use_shortcut=False):
        super(Decoder, self).__init__()
        
        self.net = nn.ModuleList()
        for i, j in pair([latent_dim] + hidden_dims):
            self.net.append(ResidualBlock(i, j, use_shortcut=use_shortcut))

        self.fc = nn.Linear(hidden_dims[-1], D_out)
        
    def forward(self, x):
        for block in self.net:
            x = block(x)
        return self.fc(x)
    
class VAE(nn.Module):
    def __init__(self, D_in, hidden_dims, latent_dim, use_shortcut=False):
        super(VAE, self).__init__()
        self.encoder = Encoder(D_in, hidden_dims, latent_dim, use_shortcut)
        self.decoder = Decoder(latent_dim, hidden_dims[::-1], D_in, use_shortcut)
        
    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # Some reasons for why we only need the mean for inference:
            # - For reconstruction, we want this to be deterministic (reason for ignoring the noise `eps`).
            # - The mean represents the most likely position in the latent space
            # - During training, the variance acts as a regularization. We can drop this during inference.
            return mu
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        return self.decoder(z), mu, logvar

class MSE_KLD(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss_KLD + loss_MSE, loss_MSE.item(), loss_KLD.item()
