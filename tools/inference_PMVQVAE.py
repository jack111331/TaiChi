import options.option_vq as option_vq
import torch
import json

args = option_vq.get_args_parser()
print("Output Directory", args.out_dir)
torch.manual_seed(args.seed)

import os
args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

args.dataname = 't2m'
args.resume_pth = 'output/VQVAE/net_last.pth'
args.down_t = 2
args.depth = 3
import numpy as np
import models.vqvae as vqvae
import warnings
warnings.filterwarnings('ignore')

assert args.code_idx_up is not None and len(args.code_idx_up) != 0
assert args.code_idx_down is not None and len(args.code_idx_down) != 0
assert len(args.code_idx_up) == len(args.code_idx_down)

code_idx_up = torch.Tensor(args.code_idx_up).to(torch.int64)
code_idx_down = torch.Tensor(args.code_idx_down).to(torch.int64)

assert torch.all(code_idx_up >= 0) and torch.all(code_idx_up < args.nb_code)
assert torch.all(code_idx_down >= 0) and torch.all(code_idx_down < args.nb_code)

code_idx = torch.cat([code_idx_up.unsqueeze(-1), code_idx_down.unsqueeze(-1)], dim=-1).unsqueeze(0)

if args.vq_resume_setting_pth is not None:
    net, _ = vqvae.SepHumanVQVAE.load_from_setting(args.vq_resume_setting_pth, ckpt_type=args.vq_resume_ckpt_type)
else:
    raise RuntimeError("PM-VQ-VAE: Doesn't provide setting file")
net.eval()
net.cuda()

mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()

code_idx = code_idx.to(net.device)
decoded_motion = net.forward_decoder(code_idx)

from utils.motion_process import recover_from_ric
decoded_joint_poses = recover_from_ric((decoded_motion * std + mean).float(), 22)
decoded_joint_poses = decoded_joint_poses.reshape(1, -1, 22, 3)

# Save the motion joints position in npy and gif formats
npy_save_path = os.path.join(args.out_dir, f"{args.out_motion_name}.npy")
gif_save_path = os.path.join(args.out_dir, f'{args.out_motion_name}.gif')
print(f"Output inference result to {npy_save_path} and {gif_save_path}")
np.save(npy_save_path, decoded_joint_poses.cpu().numpy())

import visualization.plot_3d_global as plot_3d
pose_vis = plot_3d.draw_to_batch(decoded_joint_poses.detach().cpu().numpy(),[""], [gif_save_path])