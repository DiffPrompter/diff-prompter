import math 
import torch
import torchvision 
from .gdip_vision_s import VisionEncoder

class GatedDIP(torch.nn.Module):
    """_summary_

    Args:
        torch (_type_): _description_
    """
    def __init__(self,
                encoder_output_dim : int = 256,
                num_of_gates : int = 8,
                patch_size=4):
        """_summary_

        Args:
            encoder_output_dim (int, optional): _description_. Defaults to 256.
            num_of_gates (int, optional): _description_. Defaults to 7.
        """
        super(GatedDIP,self).__init__()
        self.patch_size = patch_size
        
        self.mlp_lite = VisionEncoder(encoder_output_dim=encoder_output_dim)
        
        # Gating Module
        self.gate_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,num_of_gates,bias=True))
        
        # GAP Layer
        self.gap_layer =  torch.nn.AdaptiveAvgPool2d((1,1))
        
        # White-Balance Module
        self.wb_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,3,bias=True))
        
        # Gamma Module
        self.gamma_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))
        
        # Sharpning Module
        self.gaussian_blur = torchvision.transforms.GaussianBlur(13, sigma=(0.1, 5.0))
        self.sharpning_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))

        # De-Fogging Module
        self.defogging_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))

        # Contrast Module
        self.contrast_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))

        # Contrast Module
        self.tone_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,8,bias=True))
        
        # fft Module
        self.fft_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))

        # self.softmax = torch.nn.Softmax(dim=1)
    
    def rgb2lum(self,img: torch.tensor):
        """_summary_

        Args:
            img (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        img = 0.27 * img[:, 0, :, :] + 0.67 * img[:, 1, :,:] + 0.06 * img[:, 2, :, :]
        return img 
    
    def lerp(self ,a : int , b : int , l : torch.tensor):
        return (1 - l.unsqueeze(2).unsqueeze(3)) * a + l.unsqueeze(2).unsqueeze(3) * b


    def dark_channel(self,x : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        z = x.min(dim=1)[0].unsqueeze(1)
        return z 
    
    
    def atmospheric_light(self,x : torch.tensor,dark : torch.tensor ,top_k : int=1000):
        """_summary_

        Args:
            x (torch.tensor): _description_
            top_k (int, optional): _description_. Defaults to 1000.

        Returns:
            _type_: _description_
        """
        h,w = x.shape[2],x.shape[3]
        imsz = h * w 
        numpx = int(max(math.floor(imsz/top_k),1))
        darkvec = dark.reshape(x.shape[0],imsz,1)
        imvec = x.reshape(x.shape[0],3,imsz).transpose(1,2)
        indices = darkvec.argsort(1)
        indices = indices[:,imsz-numpx:imsz]
        atmsum = torch.zeros([x.shape[0],1,3]).to('cuda')
        for b in range(x.shape[0]):
            for ind in range(1,numpx):
                atmsum[b,:,:] = atmsum[b,:,:] + imvec[b,indices[b,ind],:]
        a = atmsum/numpx
        a = a.squeeze(1).unsqueeze(2).unsqueeze(3)
        return a
    
    def fft(self, x : torch.tensor,latent_out : torch.tensor, fft_gate : torch.tensor):
        # the smaller rate, the smoother; the larger rate, the darker
        # rate = 4, 8, 16, 32
        rate = self.fft_module(latent_out).unsqueeze(2).unsqueeze(3)
        rate = self.tanh_range(rate,torch.tensor(0.1),torch.tensor(0.8))
        rate = rate.reshape(-1,1)
        #print(rate, rate.shape)
        mask = torch.zeros(x.shape).to(x.device)
        w, h = x.shape[-2:]
        line = ((w * h * rate) ** .5 // 2).long()
        for i,l in enumerate(line):
            mask[i, :, w//2-l:w//2+l, h//2-l:h//2+l] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        # mask[fft.float() > self.freq_nums] = 1
        # high pass: 1-mask, low pass: mask
        fft = fft * (1 - mask)
        # fft = fft * mask
        fr = fft.real
        fi = fft.imag

        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real

        #print(torch.abs(inv).shape, fft_gate.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).shape)
        inv = torch.abs(inv)*(fft_gate.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))

        return inv
        
    
    def blur(self,x : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        return self.gaussian_blur(x)


    def defog(self,x:torch.tensor ,latent_out : torch.tensor ,fog_gate : torch.tensor):
        """Defogging module is used for removing the fog from the image using ASM 
        (Atmospheric Scattering Model).
        I(X) = (1-T(X)) * J(X) + T(X) * A(X)
        I(X) => image containing the fog.
        T(X) => Transmission map of the image.
        J(X) => True image Radiance.
        A(X) => Atmospheric scattering factor.

        Args:
            x (torch.tensor): Input image I(X)
            latent_out (torch.tensor): Feature representation from DIP Module.
            fog_gate (torch.tensor): Gate value raning from (0. - 1.) which enables defog module.

        Returns:
            torch.tensor : Returns defogged image with true image radiance.
        """
        omega = self.defogging_module(latent_out).unsqueeze(2).unsqueeze(3)
        omega = self.tanh_range(omega,torch.tensor(0.1),torch.tensor(1.))
        dark_i = self.dark_channel(x) 
        a = self.atmospheric_light(x,dark_i)
        i = x/a 
        i = self.dark_channel(i)
        t = 1. - (omega*i)
        j = ((x-a)/(torch.maximum(t,torch.tensor(0.01))))+a
        j = (j - j.min())/(j.max()-j.min())
        j = j* fog_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return j
        
    def white_balance(self,x : torch.tensor,latent_out : torch.tensor ,wb_gate: torch.tensor):
        """ White balance of the image is predicted using latent output of an encoder.

        Args:
            x (torch.tensor): Input RGB image.
            latent_out (torch.tensor): Output from the last layer of an encoder.
            wb_gate (torch.tensor): White-balance gate used to change the influence of color scaled image.

        Returns:
            torch.tensor: returns White-Balanced image. 
        """
        log_wb_range = 0.5
        wb = self.wb_module(latent_out)
        wb = torch.exp(self.tanh_range(wb,-log_wb_range,log_wb_range))
        
        color_scaling = 1./(1e-5 + 0.27 * wb[:, 0] + 0.67 * wb[:, 1] +
        0.06 * wb[:, 2])
        wb = color_scaling.unsqueeze(1)*wb
        wb_out = wb.unsqueeze(2).unsqueeze(3)*x
        wb_out = (wb_out-wb_out.min())/(wb_out.max()-wb_out.min())
        wb_out = wb_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)*wb_out
        return wb_out

    def tanh01(self,x : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        return torch.tanh(x)*0.5+0.5

    def tanh_range(self,x : torch.tensor,left : float,right : float):
        """_summary_

        Args:
            x (torch.tensor): _description_
            left (float): _description_
            right (float): _description_

        Returns:
            _type_: _description_
        """
        return self.tanh01(x)*(right-left)+ left

    def gamma_balance(self,x : torch.tensor,latent_out : torch.tensor,gamma_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_
            gamma_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        log_gamma = torch.log(torch.tensor(2.5))
        gamma = self.gamma_module(latent_out).unsqueeze(2).unsqueeze(3)
        gamma = torch.exp(self.tanh_range(gamma,-log_gamma,log_gamma))
        g = torch.pow(torch.maximum(x,torch.tensor(1e-4)),gamma)
        g = (g-g.min())/(g.max()-g.min())
        g = g*gamma_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return g
    
    def sharpning(self,x : torch.tensor,latent_out: torch.tensor,sharpning_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_
            sharpning_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        out_x = self.blur(x)
        y = self.sharpning_module(latent_out).unsqueeze(2).unsqueeze(3)
        y = self.tanh_range(y,torch.tensor(0.1),torch.tensor(1.))
        s = x + (y*(x-out_x))
        s = (s-s.min())/(s.max()-s.min())
        s = s * (sharpning_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3))
        return s
    
    def identity(self,x : torch.tensor,identity_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            identity_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        x = x*identity_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return x
    
    def contrast(self,x : torch.tensor,latent_out : torch.tensor,contrast_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_
            contrast_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        alpha = torch.tanh(self.contrast_module(latent_out))
        luminance = torch.minimum(torch.maximum(self.rgb2lum(x), torch.tensor(0.0)), torch.tensor(1.0)).unsqueeze(1)
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = x / (luminance + 1e-6) * contrast_lum
        contrast_image = self.lerp(x, contrast_image, alpha) 
        contrast_image = (contrast_image-contrast_image.min())/(contrast_image.max()-contrast_image.min())
        contrast_image = contrast_image * contrast_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return contrast_image
    
    def tone(self,x : torch.tensor,latent_out : torch.tensor,tone_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_
            tone_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        curve_steps = 8
        tone_curve = self.tone_module(latent_out).reshape(-1,1,curve_steps)
        tone_curve = self.tanh_range(tone_curve,0.5, 2)
        tone_curve_sum = torch.sum(tone_curve, dim=2) + 1e-30
        total_image = x * 0
        for i in range(curve_steps):
            total_image += torch.clamp(x - 1.0 * i /curve_steps, 0, 1.0 /curve_steps) \
                            * tone_curve[:,:,i].unsqueeze(2).unsqueeze(3)
        total_image *= curve_steps / tone_curve_sum.unsqueeze(2).unsqueeze(3)
        total_image = (total_image-total_image.min())/(total_image.max()-total_image.min())
        total_image = total_image * tone_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return total_image


    
    def forward(self, x : torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        emb = self.mlp_lite(x)
        ### TOGGLE ME FOR SOFTMAX
        #gate = self.tanh_range(self.gate_module(emb),0.01,1.0)
        # gate = self.softmax(self.gate_module(emb))
        gate = torch.sigmoid(self.gate_module(emb))
        
        wb_out = self.white_balance(x,emb,gate[:,0])
        gamma_out = self.gamma_balance(x,emb,gate[:,1])
        identity_out = self.identity(x,gate[:,2])
        sharpning_out = self.sharpning(x,emb,gate[:,3])
        fog_out = self.defog(x,emb,gate[:,4])
        contrast_out = self.contrast(x,emb,gate[:,5])
        tone_out = self.tone(x,emb,gate[:,6])
        fft_out  = self.fft(x,emb,gate[:,7])
        x = wb_out + gamma_out   + fog_out + sharpning_out + contrast_out + tone_out + identity_out + fft_out
        x = (x-x.min())/(x.max()-x.min())
        return x,emb,gate

if __name__ == '__main__':
    patch_size = 16 # patch_size
    image_size = 1024
    num_patches = image_size//patch_size
    encoder_out_dim = 1280
    
    # dummy embedding feature
    emb = torch.randn(11,num_patches,num_patches,encoder_out_dim).cuda()
    
    # dummy imput image
    x = torch.randn(11,3,image_size,image_size).cuda()
    x = (x-x.min())/(x.max()-x.min())
    
    # Initialize model
    model = GatedDIP().cuda()
    print(model)
    
    # Dummy inference
    out,gate= model(x,emb)
    print('out shape:',out.shape)
    print('gate shape:',gate.shape)
