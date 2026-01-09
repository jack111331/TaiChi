import os 
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import clip

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_eval
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')

##### ---- Exp dirs ---- #####
args, _ = option_trans.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer)

wrapper_opt = get_opt(args.dataname, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####

## load clip model and datasets
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

clip_output_wrapper = trans.CLIPOutputWrapper(clip_model)

if True:
    if args.vq_resume_setting_pth is not None:
        net, _ = vqvae.SepHumanVQVAE.load_from_setting(args.vq_resume_setting_pth, ckpt_type='best_fid')
    else:
        raise RuntimeError("PM-VQ-VAE: Doesn't provide setting file")

    net.eval()
    net.cuda()

if True:
    if args.lptgpt_resume_setting_pth is not None:
        trans_encoder = trans.Text2Motion_Transformer_Word_CrossAtt.load_from_setting(args.lptgpt_resume_setting_pth, ckpt_type='best_fid')
    else:
        raise RuntimeError("LPT-GPT: Doesn't provide setting file")

    trans_encoder.eval()
    trans_encoder.cuda()


fid = []
div = []
top1 = []
top2 = []
top3 = []
mm_dist = []
multi = []
repeat_time = 20

        
for i in range(repeat_time):
    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_mm_dist, best_multi, writer, logger = eval_trans.evaluation_transformer_test(args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_mm_dist=100, best_multi=0, clip_output_wrapper=clip_output_wrapper, eval_wrapper=eval_wrapper, cfg_scale=args.cfg_scale, nucleus_p=args.nucleus_p, draw=False, savegif=False, save=False, savenpy=(i==0))
    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    mm_dist.append(best_mm_dist)
    multi.append(best_multi)

print('final result:')
print('fid: ', sum(fid)/repeat_time)
print('div: ', sum(div)/repeat_time)
print('top1: ', sum(top1)/repeat_time)
print('top2: ', sum(top2)/repeat_time)
print('top3: ', sum(top3)/repeat_time)
print('mm_dist: ', sum(mm_dist)/repeat_time)
print('multi: ', sum(multi)/repeat_time)

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
mm_dist = np.array(mm_dist)
multi = np.array(multi)
msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, mm_dist. {np.mean(mm_dist):.3f}, conf. {np.std(mm_dist)*1.96/np.sqrt(repeat_time):.3f}, Multi. {np.mean(multi):.3f}, conf. {np.std(multi)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)