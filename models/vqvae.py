import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
import torch
import json
import os

t2m_kinematic_up = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
t2m_kinematic_down = [1, 2, 4, 5, 7, 8, 10, 11]
kit_kinematic_up = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 21]
kit_kinematic_down = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

class VQVAE_part_no_decoder(nn.Module):
    def __init__(self,
                 input_dim=3,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 quantizer='ema_reset',
                 **kwargs):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = quantizer
        self.encoder = Encoder(input_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        if quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, kwargs["mu"])
        elif quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim)


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = x_encoder.permute(0, 2, 1)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx

    def forward(self, x):
        
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        
        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)
        
        return x_quantized, loss, perplexity
    
    def dequantize(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        return x_d
        
class SepHumanVQVAE(nn.Module):
    def __init__(self,
                 dataname,
                 nb_code_up=512,
                 nb_code_down=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 joint_emb_dim=32,
                 **kwargs):
        super().__init__()
        
        if dataname == 'kit':
            self.nb_joints = 21
            self.kinematic_up = kit_kinematic_up
            self.kinematic_down = kit_kinematic_down
            total_joints = 251
            self.decompose_joints_f = self.decompose_joints_kit

        elif dataname == 't2m':
            self.nb_joints = 22
            self.kinematic_up = t2m_kinematic_up
            self.kinematic_down = t2m_kinematic_down
            total_joints = 263
            self.decompose_joints_f = self.decompose_joints_t2m
        
        else:
            raise NotImplementedError
        
        assert kwargs.get("quantizer_setting", None) is not None
        
        self.joint_emb_dim = joint_emb_dim
        
        self.vqvae_up = VQVAE_part_no_decoder(len(self.kinematic_up) * self.joint_emb_dim, nb_code_up, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, **kwargs["quantizer_setting"])
        self.vqvae_down = VQVAE_part_no_decoder(len(self.kinematic_down) * self.joint_emb_dim, nb_code_down, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, **kwargs["quantizer_setting"])
        self.decoder = Decoder(total_joints, output_emb_width * 2, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

        self.root_joint_embed = nn.Linear(7, self.joint_emb_dim)
        self.other_joint_embed = nn.Linear(12, self.joint_emb_dim)
        self.contact_embed = nn.Linear(4, self.joint_emb_dim)

    # HumanML3D parse motion representation: https://github.com/EricGuo5513/HumanML3D/blob/main/motion_representation.ipynb
    # Adopted code from https://github.com/ZcyMonkey/AttT2M/blob/main/models/encdec.py
    def decompose_joints_t2m(self, x):
        b, t, c = x.size()
        root_feature = torch.cat((x[:,:,:4],x[:,:,193:196]),dim=2).unsqueeze(2) # [b, t, 1, 7], 7 dimensions represent: [rotation velocity along y-axis (1, ), linear velocity on xz plane (2, ), root height (1, ), velocity (3, )]
        other_joint = torch.cat((x[:,:,4:193],x[:,:,196:259]),dim=2)
        position = other_joint[:,:,:63].reshape(b, t, 21, 3)
        rotation = other_joint[:,:,63:189].reshape(b, t, 21, 6)
        velocity = other_joint[:,:,189:].reshape(b, t, 21, 3)
        other_joint_feature = torch.cat([position, rotation, velocity], dim=3)
        contact = x[:,:,259:].unsqueeze(2) # [b, t, 1, 4]
        return root_feature, other_joint_feature, contact

    def decompose_joints_kit(self, x):
        b, t, c = x.size()
        root_feature = torch.cat((x[:,:,:4],x[:,:,184:187]),dim=2).unsqueeze(2) #  # [b, t, 1, 7], 7 dimensions represent: [rotation velocity along y-axis (1, ), linear velocity on xz plane (2, ), root height (1, ), velocity (3, )]
        other_joint = torch.cat((x[:,:,4:184],x[:,:,187:247]),dim=2)
        position = other_joint[:,:,:60].reshape(b, t, 20, 3)
        rotation = other_joint[:,:,60:180].reshape(b, t, 20, 6)
        velocity = other_joint[:,:,180:].reshape(b, t, 20, 3)
        other_joint_feature = torch.cat([position, rotation, velocity], dim=3)
        contact = x[:,:,247:].unsqueeze(2) # [b, t, 1, self.joint_emb_dim]
        return root_feature, other_joint_feature, contact
            
    def divide_to_up_down(self, x):
        b, t, c = x.size()
        root_feature, other_joint_feature, contact = self.decompose_joints_f(x)        

        # Embed feature
        root_feature = self.root_joint_embed(root_feature)
        other_joint_feature = self.other_joint_embed(other_joint_feature)
        contact_feature = self.contact_embed(contact)
        x_full_body = torch.cat((root_feature, other_joint_feature, contact_feature), dim=2) # [b, t, 23(root: 1, other joints: 21, contact: 1), self.joint_emb_dim]
        x_up = x_full_body[:, :, self.kinematic_up].reshape(b, t, -1)
        x_down = x_full_body[:, :, self.kinematic_down].reshape(b, t, -1)
        return x_up, x_down
    
    def encode(self, x):
        x_up, x_down = self.divide_to_up_down(x)
        quants_up = self.vqvae_up.encode(x_up) # [b, T]
        quants_down = self.vqvae_down.encode(x_down) # [b, T]
        return (quants_up, quants_down)

    def forward(self, x):
        x = x.float()
        x_up, x_down = self.divide_to_up_down(x)
        x_quantized_up, loss_up, perplexity_up = self.vqvae_up(x_up)
        x_quantized_down, loss_down, perplexity_down = self.vqvae_down(x_down)
        x_out = self.decoder(torch.cat([x_quantized_up, x_quantized_down], dim=1)).permute(0, 2, 1) # [B, F, T] -> [B, T, F]
        return x_out, (loss_up + loss_down) * 0.5, (perplexity_up, perplexity_down)

    def forward_decoder(self, x):
        x_up, x_down = x
        x_quantized_up = self.vqvae_up.dequantize(x_up) # [1, code_dim, B*T]
        x_quantized_down = self.vqvae_down.dequantize(x_down) # [1, code_dim, B*T]
        x_out = self.decoder(torch.cat([x_quantized_up, x_quantized_down], dim=1)).permute(0, 2, 1) # [B, F, T] -> [B, T, F]
        return x_out
    
    @staticmethod
    def load_from_setting(setting_path, ckpt_type):
        with open(setting_path, 'r') as f:
            vqvae_settings = json.load(f)
            
        # Complement relpath to abspath
        if ckpt_type in vqvae_settings['checkpoint_dict']:
            ckpt_path = os.path.join(os.path.dirname(setting_path), vqvae_settings["checkpoint_dir"], vqvae_settings["checkpoint_dict"][ckpt_type])
        else:
            raise ValueError(f"LPT-GPT: Checkpoint type {ckpt_type} not found.")

        vqvae_settings.pop("checkpoint_dir", None)
        vqvae_settings.pop("checkpoint_dict")
        if os.path.exists(ckpt_path):
            print ('loading PM-VQ-VAE checkpoint from {}'.format(ckpt_path))
            vqvae = SepHumanVQVAE(**vqvae_settings)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            vqvae.load_state_dict(ckpt['net'], strict=True)
        
        else:
            raise RuntimeError(f"PM-VQ-VAE: Checkpoint path {ckpt_path} not found")
        
        return vqvae, vqvae_settings