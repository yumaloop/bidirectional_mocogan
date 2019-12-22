import os
import numpy as np
import torch
from torch import nn

def to_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)

class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super(Noise, self).__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * to_cuda(torch.FloatTensor(x.size()).normal_())
        return x

    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
    
class ImageEncoder(nn.Module):
    """
    Z'm = E(X)
    Image Encoder: X → Z'm
    """
    def __init__(self, 
                 n_channels, 
                 dim_z_motion, 
                 ndf=32, 
                 use_noise=False, 
                 noise_sigma=None):
        
        super(ImageEncoder, self).__init__()

        self.n_channels = n_channels
        self.dim_z_motion = dim_z_motion
        self.use_noise = use_noise
        self.noise_sigma = noise_sigma

        self.infer_image = nn.Sequential(
            Noise(self.use_noise, sigma=self.noise_sigma),
            nn.Conv2d(self.n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(self.use_noise, sigma=self.noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(self.use_noise, sigma=self.noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(self.use_noise, sigma=self.noise_sigma),
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            
            Flatten(),
            nn.Linear(16, self.dim_z_motion),
             # nn.Tanh(),
        )

    def forward(self, x):
        """
        x: 5D-Tensor (batch_size, channel, video_len, height, width)
        """
        x = x.permute(2, 1, 0, 3, 4) # x: (video_len, channel, batch_size. height, width)        
        zm = []
        for x_t in x:
            x_t = x_t.permute(1, 0, 2, 3) # x_t: (batch_size, channel, height, width)
            zm_t = self.infer_image(x_t)  # zm_t: (batch_size, dim_z_motion)
            zm.append(zm_t)
        zm = torch.stack(zm)
        zm = zm.permute(1, 0, 2)
        zm = to_cuda(zm)
        # zm: 3D-Tensor (batch_size, video_len, dim_z_motion)
        return zm
    
    
class VideoGenerator(nn.Module):
    """
    X' = G(Zc, Zm)
    Video Generator: (Zc, Zm) → X'
    """
    def __init__(self, 
                 n_channels, 
                 dim_z_content, 
                 dim_z_motion,
                 video_length=20, 
                 ngf=32,
                 use_noise=False, 
                 noise_sigma=None):
        
        super(VideoGenerator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_motion = dim_z_motion
        self.dim_z = dim_z_motion + dim_z_content
        self.video_length = video_length
        
        # GRU
        self.recurrent = nn.GRUCell(dim_z_motion, dim_z_motion)

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.ConvTranspose2d(self.dim_z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf, self.n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def sample_z_motion(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length
        h_t = [self.get_gru_initial_state(num_samples)]

        for frame_num in range(video_len):
            e_t = self.get_iteration_noise(num_samples)
            h_t.append(self.recurrent(e_t, h_t[-1]))
        
        # make z_m
        z_m_t = [h_k.view(-1, 1, self.dim_z_motion) for h_k in h_t]
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, video_len, self.dim_z_motion)
        z_m = to_cuda(z_m) # z_m: (num_samples, video_len, dim_z_motion)
        return z_m

    def sample_z_content(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length
        content = np.random.normal(0, 1, (num_samples, self.dim_z_content)).astype(np.float32) # (10,1)
        content = np.tile(content, (video_len, 1, 1)) # (20, 10, 1)
        content = np.transpose(content, (1,0,2))                
        z_c = torch.FloatTensor(content)
        z_c = to_cuda(z_c) # z_c: (num_samples, video_len, dim_z_content)
        return z_c

    def sample_z(self, num_samples, video_len=None):
        z_c = self.sample_z_content(num_samples, video_len) # (batch_size, video_len, dim_z_content)
        z_m = self.sample_z_motion(num_samples, video_len) # (batch_size, video_len, dim_z_motion)
        z = torch.cat((z_c, z_m), dim=2)
        z = to_cuda(z)
        return z, z_c, z_m
    
    def sample_videos(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length
        z, z_c, z_m = self.sample_z(num_samples, video_len)
        h = self.main(z.view(int(z.size(0)*z.size(1)), z.size(2), 1, 1))
        h = h.view(int(h.size(0) / video_len), video_len, self.n_channels, h.size(3), h.size(3))
        h = h.permute(0, 2, 1, 3, 4)
        h = to_cuda(h)
        return h, z, z_c, z_m

    def sample_images(self, num_samples):
        z, z_c, z_m = self.sample_z(num_samples, video_len=1)
        h = self.main(z.view(int(z.size(0)*z.size(1)), z.size(2), 1, 1))
        h = to_cuda(h)
        z   = torch.squeeze(z)
        z_c = torch.squeeze(z_c)
        z_m = torch.squeeze(z_m)
        return h, z, z_c, z_m
    
    def sample_images_and_videos(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length
        z, z_c, z_m = self.sample_z(num_samples)  
        
        # fake images
        h = self.main(z.view(int(z.size(0)*z.size(1)), z.size(2), 1, 1))
        images_fake = to_cuda(h)
        
        # fake videos
        h = self.main(z.view(int(z.size(0)*z.size(1)), z.size(2), 1, 1))
        h = h.view(int(h.size(0) / video_len), video_len, self.n_channels, h.size(3), h.size(3))
        h = h.permute(0, 2, 1, 3, 4)
        videos_fake = to_cuda(h)

        return images_fake, videos_fake, z, z_c, z_m

    def get_gru_initial_state(self, num_samples):
        # Random values following standard gauss
        return to_cuda(torch.FloatTensor(num_samples, self.dim_z_motion).normal_())

    def get_iteration_noise(self, num_samples):
        # Random values following standard gauss
        return to_cuda(torch.FloatTensor(num_samples, self.dim_z_motion).normal_())


class ImageDiscriminator(nn.Module):
    """
    {1, 0} = DI(X)
    Image Discriminator: {X[i], X'[i]} → {1, 0}
    """
    def __init__(self, 
                 n_channels, 
                 dim_z_content,
                 dim_z_motion,
                 dropout,                 
                 ndf=32, 
                 video_length=20, 
                 use_noise=False, 
                 noise_sigma=None):
        
        super(ImageDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_motion = dim_z_motion
        self.dim_z = dim_z_motion + dim_z_content
        self.dropout = dropout
        self.video_length = video_length
        self.use_noise = use_noise

        self.infer_x = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Flatten(),
            nn.Linear(ndf * 4 * 4 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=self.dropout)
        )
        
        self.infer_zm = nn.Sequential(
            nn.Linear(self.dim_z_motion, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=self.dropout),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=self.dropout)
        )

        self.infer_joint = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=self.dropout),

            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x, zm):
        """
        x: 4D-Tensor (num_images, channel, height, width)
        zm: 2D-Tensor (num_images, dim_z_motion)
        """
        output_x = self.infer_x(x) # (num_images, 512)
        output_zm = self.infer_zm(zm) # (num_images, 512)
        output_x_zm = torch.cat([output_x, output_zm], dim=1) # (num_images, 1024)
        output = self.infer_joint(output_x_zm)
        output = output.squeeze()
        output = to_cuda(output)
        return output
    

class VideoDiscriminator(nn.Module):
    """
    {1, 0} = DV(X)
    Video Discriminator: {X, X'} → {1, 0}
    """
    def __init__(self, 
                 n_channels, 
                 dim_z_content,
                 dim_z_motion,
                 dropout,
                 video_length=20,
                 n_output_neurons=1, 
                 bn_use_gamma=True, 
                 use_noise=False, 
                 noise_sigma=None, 
                 ndf=32):
        
        super(VideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_motion = dim_z_motion
        self.dim_z = dim_z_motion + dim_z_content
        self.dropout = dropout
        self.n_output_neurons = n_output_neurons
        self.video_length = video_length
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        # input_shape : (batch_size, ch_in, duration, height, width)
        self.infer_x = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 4, ndf * 8, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            Flatten(),
            nn.Linear(ndf * 4 * 4 * 4 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=self.dropout)
        )
        
        self.infer_zm = nn.Sequential(
            nn.Linear(self.dim_z_motion * self.video_length, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=self.dropout),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=self.dropout)
        )

        self.infer_joint = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=self.dropout),

            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, zm):
        """
        x: 5D-Tensor (batch_size, video_len, channel, height, widt h)
        zm: 3D-Tensor (batch_size, video_len, dim_z_motion)
        """
        zm = zm.contiguous().view(zm.size(0), int(zm.size(1)*zm.size(2))) # zm: (batch_size, dim_z_motion*video_length)
        
        output_x = self.infer_x(x) # (batch_size, 512)
        output_zm = self.infer_zm(zm) # (batch_size, 512)
        
        output_x_zm = torch.cat([output_x, output_zm], dim=1)
        output = self.infer_joint(output_x_zm)
        output = output.squeeze()
        output = to_cuda(output)
        return output
