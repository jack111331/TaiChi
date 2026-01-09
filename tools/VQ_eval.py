import os
import json

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import models.vqvae as vqvae
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_TM_eval
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
import numpy as np
##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')


wrapper_opt = get_opt(args.dataname, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Dataloader ---- #####
args.nb_joints = 21 if args.dataname == 'kit' else 22

val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer, unit_length=2**args.down_t)

##### ---- Network ---- #####
if args.vq_resume_setting_pth is not None:
    net, _ = vqvae.SepHumanVQVAE.load_from_setting(args.vq_resume_setting_pth, ckpt_type=args.vq_resume_ckpt_type)
else:
    raise RuntimeError("PM-VQ-VAE: Doesn't provide setting file")

net.eval()
net.cuda()

fid = []
div = []
top1 = []
top2 = []
top3 = []
mm_dist = []
mpjpe = []
repeat_time = 20
for i in range(repeat_time):
    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_mm_dist, best_mpjpe, writer, logger = eval_trans.evaluation_vqvae(args.out_dir, args.out_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_mm_dist=100, best_mpjpe=1000, eval_wrapper=eval_wrapper, draw=False, save=False, savenpy=(i==0))
    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    mm_dist.append(best_mm_dist)
    mpjpe.append(best_mpjpe)
print('final result:')
print('fid: ', sum(fid)/repeat_time)
print('div: ', sum(div)/repeat_time)
print('top1: ', sum(top1)/repeat_time)
print('top2: ', sum(top2)/repeat_time)
print('top3: ', sum(top3)/repeat_time)
print('MM-Dist: ', sum(mm_dist)/repeat_time)
print('MPJPE: ', sum(mpjpe)/repeat_time)

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
mm_dist = np.array(mm_dist)
mpjpe = np.array(mpjpe)
msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, MM-Dist. {np.mean(mm_dist):.3f}, conf. {np.std(mm_dist)*1.96/np.sqrt(repeat_time):.3f}, MPJPE. {np.mean(mpjpe):.3f}, conf. {np.std(mpjpe)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)