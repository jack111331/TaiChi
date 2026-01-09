from utils.render_tool import render_smpl_mesh, render_smpl_mesh_still, render_skeleton, body_part_dict
from visualization.joints2bvh import Joint2BVHConvertor
import numpy as np
import imageio
from os.path import join as opjoin
from pathlib import Path
from os import makedirs

def load_motions(input_dir, input_pth_list, reconstruction=False):
    import torch
    import numpy as np
    
    humanml3d_mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
    humanml3d_std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()

    for idx, input_motion_pth in enumerate(input_pth_list):
        input_motion = np.load(opjoin(input_dir, input_motion_pth))
        poses = None
        # Preprocess input motions
        if input_motion.shape[-1] == 263:
            # HumanML3D Dataset preprocessed motions
            # [frames, 263]
            from utils.motion_process import recover_from_ric
            if reconstruction == False:
                poses = input_motion[None, ...]
                input_motion = recover_from_ric((torch.from_numpy(input_motion).cuda().unsqueeze(0)).float(), 22)
                input_motion = input_motion.reshape(1, -1, 22, 3)
                input_motion = input_motion.detach().cpu().numpy()
            else:
                import options.option_transformer as option_trans
                import torch
                from options.get_eval_option import get_opt

                args, _ = option_trans.get_args_parser()
                print("Output Directory", args.out_dir)
                torch.manual_seed(args.seed)

                import os
                args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
                os.makedirs(args.out_dir, exist_ok = True)

                args.dataname = 't2m'
                # Original pretrained ckpt
                # args.resume_pth = 'pretrained/VQVAE/net_last.pth'
                # args.resume_trans = 'pretrained/VQTransformer_corruption05/net_best_fid.pth'
                # Our reproduce ckpt
                args.down_t = 2
                args.depth = 3
                args.block_size = 51
                import clip
                import numpy as np
                import models.vqvae as vqvae
                import models.t2m_trans as trans
                import warnings
                warnings.filterwarnings('ignore')

                if True:
                    if args.vq_resume_setting_pth is not None:
                        net, _ = vqvae.SepHumanVQVAE.load_from_setting(args.vq_resume_setting_pth, ckpt_type='best_fid')
                    else:
                        raise RuntimeError("PM-VQ-VAE: Doesn't provide setting file")

                    net.eval()
                    net.cuda()

                m_length = input_motion.shape[0]

                input_motion = torch.Tensor(input_motion).float().cuda()[None, ...]
                input_motion, loss_commit, perplexity = net((input_motion - humanml3d_mean) / humanml3d_std)

                poses = input_motion # Batch
                input_motion = recover_from_ric((input_motion * humanml3d_std + humanml3d_mean).float(), 22)
                input_motion = input_motion.reshape(1, -1, 22, 3)
                input_motion = input_motion.detach().cpu().numpy()

        elif input_motion.shape[-1] == 3 and input_motion.shape[-2] == 22:        
            pass

        elif input_motion.shape[-1] == 252:
            # KIT-ML Dataset preprocessed motions
            raise NotImplementedError

        else:
            raise RuntimeError("Input motion format not support.")

        # int, [1, frames, 22, 3], [1, frames, 263]
        yield idx, input_motion, poses

def generate_motions(text_prompts):
    # Prepare models to generate motions
    if True:
        import options.option_transformer as option_trans
        import torch
        from options.get_eval_option import get_opt

        args, _ = option_trans.get_args_parser()
        print("Output Directory", args.out_dir)
        torch.manual_seed(args.seed)

        import os
        args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
        os.makedirs(args.out_dir, exist_ok = True)

        args.dataname = 't2m'
        # Original pretrained ckpt
        # args.resume_pth = 'pretrained/VQVAE/net_last.pth'
        # args.resume_trans = 'pretrained/VQTransformer_corruption05/net_best_fid.pth'
        # Our reproduce ckpt
        args.down_t = 2
        args.depth = 3
        args.block_size = 51
        import clip
        import numpy as np
        import models.vqvae as vqvae
        import models.t2m_trans as trans
        import warnings
        warnings.filterwarnings('ignore')

        ## load clip model and datasets
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False, download_root='./')  # Must set jit=False for training
        clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        
        clip_output_wrapper = trans.CLIPOutputWrapper(clip_model)

        # Initialize pre-trained models
        if True:
            if args.vq_resume_setting_pth is not None:
                net, _ = vqvae.SepHumanVQVAE.load_from_setting(args.vq_resume_setting_pth, ckpt_type='best_fid')
            else:
                raise RuntimeError("PM-VQ-VAE: Doesn't provide setting file")

            net.eval()
            net.cuda()

            if args.lptgpt_resume_setting_pth is not None:
                trans_encoder = trans.Text2Motion_Transformer_Word_CrossAtt.load_from_setting(args.lptgpt_resume_setting_pth, ckpt_type='best_fid')
            else:
                raise RuntimeError("LPT-GPT: Doesn't provide setting file")

            trans_encoder.eval()
            trans_encoder.cuda()

        mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
        std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()
    for idx, text_prompt in enumerate(text_prompts):
        sent_feature, word_feature, word_length = clip_output_wrapper(text_prompt)
        # Generate motion from condition
        if True:
            index_motion_up, index_motion_down, pred_token_len = trans_encoder.sample_fast_nucleus(sent_feature, word_feature, word_length, cfg_scale=3.0, if_categorial=True)
            index_motion_up_k = index_motion_up[0:1, :pred_token_len[0]]
            index_motion_down_k = index_motion_down[0:1, :pred_token_len[0]]
            pred_pose = net.forward_decoder((index_motion_up_k[..., None], index_motion_down_k[..., None]))

        from utils.motion_process import recover_from_ric
        from scipy.ndimage import gaussian_filter
        def motion_feature_temporal_filter(motion, sigma=1):
            for i in range(motion.shape[1]):
                motion[:, i] = gaussian_filter(motion[:, i],
                                            sigma=sigma,
                                            mode="nearest")
            return motion

        def smooth_humanml_part(motion, sigma=1):
            def get_pos_rot_vel_idx(joint_idx):
                return [j for j in range(4 + joint_idx * 3, 4 + joint_idx * 3 + 3)] + [j for j in range(4 + 63 + joint_idx * 6, 4 + 63 + joint_idx * 6 + 6)] + [j for j in range(196 + joint_idx * 3, 196 + joint_idx * 3 + 3)]
            smoothed_motion = np.array(motion)
            # target_joint_idx = get_pos_rot_vel_idx(3) + get_pos_rot_vel_idx(6) + get_pos_rot_vel_idx(9) + get_pos_rot_vel_idx(12) + get_pos_rot_vel_idx(15) # Spine1 + Spine2 + neck + Head
            target_joint_idx = [j for j in range(0, 4)] + [j for j in range(193, 196)] + get_pos_rot_vel_idx(3-1) + get_pos_rot_vel_idx(6-1) + get_pos_rot_vel_idx(9-1) + get_pos_rot_vel_idx(12-1) + get_pos_rot_vel_idx(15-1) # Spine2 + neck
            # Lower-body
            # target_joint_idx += get_pos_rot_vel_idx(1-1) + get_pos_rot_vel_idx(2-1) + get_pos_rot_vel_idx(4-1) + get_pos_rot_vel_idx(5-1) + get_pos_rot_vel_idx(7-1) + get_pos_rot_vel_idx(8-1) + get_pos_rot_vel_idx(10-1) + get_pos_rot_vel_idx(11-1) # LR Hips + LR Knee + LR ankle + LR foot
            # Upper-body
            # target_joint_idx += get_pos_rot_vel_idx(13-1) + get_pos_rot_vel_idx(14-1) + get_pos_rot_vel_idx(16-1) + get_pos_rot_vel_idx(17-1) + get_pos_rot_vel_idx(18-1) + get_pos_rot_vel_idx(19-1) + get_pos_rot_vel_idx(20-1) + get_pos_rot_vel_idx(21-1) # LR Collar + LR Shoulder + LR elbow + LR wrist
            smoothed_motion[:, target_joint_idx] = motion_feature_temporal_filter(motion[:, target_joint_idx], sigma)
            # smoothed_motion[:, :259] = motion_feature_temporal_filter(motion[:, :259], sigma)
            return smoothed_motion
        
        pred_pose = torch.from_numpy(smooth_humanml_part(pred_pose[0].detach().cpu().numpy())).cuda()[None, ...]        
        pred_xyz = recover_from_ric((pred_pose*std+mean).float(), 22)
        xyz = pred_xyz.reshape(1, -1, 22, 3)
        yield idx, xyz.detach().cpu().numpy(), pred_pose.detach().cpu().numpy()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Input arguments
    parser.add_argument("--texts", type=str, nargs="+", default=None, help='Text prompt')
    parser.add_argument("--input-dir", type=str, default="", help='Input motion directory')
    parser.add_argument("--input-pth", type=str, nargs="+", default=None, help='Input motion filename (.npy)')

    # Render arguments
    parser.add_argument("--render-form", type=str, default="smpl-mesh", choices=['smpl-mesh', 'smpl-mesh-still', 'skeleton', 'bvh'])
    parser.add_argument("--render-frames", action='store_true', help="Render individual frames into images")
    parser.add_argument("--skeleton-bodypart", type=str, default='full-body', choices=[bodypart for bodypart in body_part_dict.keys()])
    parser.add_argument('--with-floor', action='store_true', help="Render floor")
    parser.add_argument('--with-trajectory', action='store_true', help="Render trajectory")
    parser.add_argument("--reconstruction", action='store_true', help="Render reconstruction")

    # Output arguments
    parser.add_argument('--output-dir', default="render_visualization", type=str, help="Output dir")
    parser.add_argument('--output-fn', default=None, nargs="+", type=str, help="motion name list")
    parser.add_argument('--output-format', default="gif", type=str, choices=['gif', 'mp4', 'png', 'jpg', 'bvh'], help="output format")
    parser.add_argument('--fps', default=20, type=int, help="motion name list")
    args, _ = parser.parse_known_args()
    
    print(args.texts, args.output_fn)
    
    if args.texts is not None and args.input_pth is not None:
        raise ValueError("Input approach can only be either text prompt or motion.")
    
    if args.render_frames and args.output_format not in ["png", "jpg"]:
        raise ValueError(f"Output format '{args.output_format}' isn't image given rendering individual frames (--render-frames).")
    elif not args.render_frames and args.output_format not in ["gif", "mp4", "bvh"]:
        raise ValueError(f"Output format '{args.output_format}' isn't videos, animated images, or bvh.")
        
    motion_generator = None

    if args.texts:
        print(args.texts)
        if len(args.texts) != len(args.output_fn):
            raise ValueError("Mismatch amount of text prompts to output filenames.")
        
        motion_generator = generate_motions(args.texts)

    elif args.input_pth:
        if len(args.input_pth) != len(args.output_fn):
            raise ValueError("Mismatch amount of input motions to output filenames.")

        motion_generator = load_motions(args.input_dir, args.input_pth, reconstruction=args.reconstruction)
        
    if motion_generator == None:
        raise ValueError("Please provide input which forms in either text prompt (--text) or motion (--input-pth).")
        
    makedirs(args.output_dir, exist_ok=True)
    extension_name = "." + args.output_format
    
    scene_setup = {"with_floor": args.with_floor, "with_trajectory": args.with_trajectory}
    
    for idx, motions, poses in motion_generator:
        # from scipy.ndimage import gaussian_filter
        # def motion_temporal_filter(motion, sigma=1):
        #     motion = motion.reshape(motion.shape[0], -1)
        #     for i in range(motion.shape[1]):
        #         motion[:, i] = gaussian_filter(motion[:, i],
        #                                     sigma=sigma,
        #                                     mode="nearest")
        #     return motion.reshape(motion.shape[0], -1, 3)
        # motions = motion_temporal_filter(motions[0])[None, ...]
        
        
        frame_lst = None
        if args.render_form == "smpl-mesh":
            frame_lst = render_smpl_mesh(motions[0], poses[0], **scene_setup)
        if args.render_form == "smpl-mesh-still":
            frame_lst = render_smpl_mesh_still(motions[0], None, **scene_setup)
        elif args.render_form == "skeleton":
            frame_lst = render_skeleton(motions[0], poses[0], body_part=args.skeleton_bodypart, **scene_setup)
        elif args.render_form == "bvh":
            converter = Joint2BVHConvertor()
            # np.save("test_motion.npy", motions[0])
            _, bvh_ik_joint = converter.convert(motions[0], filename=opjoin(args.output_dir, args.output_fn[idx] + "_ik.bvh"), iterations=100)
            _, bvh_joint = converter.convert(motions[0], filename=opjoin(args.output_dir, args.output_fn[idx] + ".bvh"), iterations=100, foot_ik=False)
            
        # Easy workaround of detect if we should export render results or bvh joints
        if frame_lst is not None:
            if args.render_frames:
                makedirs(opjoin(args.output_dir, f"{args.output_fn[idx]}"), exist_ok=True)
                n_frames = len(frame_lst)
                for frame_idx in range(n_frames):
                    imageio.imsave(opjoin(args.output_dir, f"{args.output_fn[idx]}", f"{frame_idx:03d}{extension_name}"), frame_lst[frame_idx])
            else:
                imageio.mimsave(opjoin(args.output_dir, args.output_fn[idx] + extension_name), frame_lst, duration=(1000 * 1 / args.fps), loop=0)
