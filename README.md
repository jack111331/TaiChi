# Tai-Chi: Text-to-Motion Generation with Locality-Aware Bipartite Body-Part Motion Prior
## Abstract
With rising demands in animation production, text-to-motion generation has become essential for the film and gaming industries. Yet, generating high-fidelity motions that seamlessly align with intended semantic meanings remains an open challenge.
To address this issue and effectively model diverse human activities, we introduce a part-decomposed motion vector quantised-variational autoencoder (PM-VQ-VAE), which encodes upper- and lower-body motions separately into discrete codebooks—forming a bipartite body-part motion prior that enables more precise representation of individual body-part movements. During decoding, PM-VQ-VAE explicitly considers the coordination between upper- and lower-body parts, enabling the synthesis of high-fidelity and varied full-body motions.
Next, we propose a locality-aware part-based text-to-motion generative pretrained transformer (LPT-GPT), which integrates locality-aware part-based attention and a learnable text attention bias. This design strengthens temporal relationships among body parts, enhances textual guidance, aligns motion with text semantics, and autoregressively predicts body-part motion code indices from a given text prompt. These code indices are used to retrieve motion codes from the PM-VQ-VAE codebooks, which are then decoded into full-body motion. Despite its conceptual simplicity, our LPT-GPT effectively predicts body-part motion code indices for the PM-VQ-VAE, facilitating the generation of high-fidelity full-body motions that accurately embed textual semantics. Our approach outperforms state-of-the-art approaches on both the HumanML3D and KIT-ML benchmark datasets, achieving superior performance with fewer network parameters. We further demonstrate the effectiveness of our approach in enabling flexible part-based motion composition.

# Install
## Environment and checkpoints
```=shell
conda env create -f environment.yml
conda activate TaiChi
huggingface-cli download isEdge/Taichi-ckpt --local-dir output/
```
## Dependencies
Please follow the **Installation** section from [T2M-GPT](https://github.com/Mael-zys/T2M-GPT) to install Glove, SMPL base model, evaluation datasets, and motion & text feature extractors.

# Evaluation
After configured dependencies section, run the following codes to produce PM-VQ-VAE and LPT-GPT's evaluation results.
## PM-VQ-VAE evaluation
```
export DATASET_DIR=<Absolute path to evaluation dataset>
python3 -m tools.VQ_eval \
--batch-size 256 \
--lr 2e-4 \
--total-iter 300000 \
--lr-scheduler 200000 \
--nb-code 512 \
--down-t 2 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir output \
--dataname t2m \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name TEST_PMVQVAE \
--vq-resume-setting-pth output/PMVQVAE/vq_setting.json
```

## LPT-GPT evaluation
```
export DATASET_DIR=<Absolute path to evaluation dataset>
python3 -m tools.GPT_eval_multi  \
--exp-name TEST_LPTGPT \
--block-sliding-window 5 \
--batch-size 128 \
--num-layers 9 \
--embed-dim-gpt 336 \
--nb-code 512 \
--n-head-gpt 8 \
--block-size 51 \
--ff-rate 2 \
--drop-out-rate 0.1 \
--vq-name PMVQVAE \
--out_dir output \
--total-iter 300000 \
--lr-scheduler 20000 \
--lr 0.0001 \
--gamma 0.1 \
--dataname t2m \
--down-t 2 \
--depth 3 \
--quantizer ema_reset \
--eval-iter 1000 \
--pkeep 0.5 \
--dilation-growth-rate 3 \
--vq-act relu \
--vq-resume-setting-pth output/PMVQVAE/vq_setting.json \
--lptgpt-resume-setting-pth output/LPTGPT/lpt_gpt_setting.json
```

# Demo
To render generation results in skeleton animated images, run the following command.
```
python -m tools.render_animate \
    --seed 122 \
    --texts "a person walks forward and gets pushed, he stumbles to the left and then returns to his original path" \
    --render-form skeleton \
    --with-floor \
    --output-dir render_gif \
    --output-fn "pushed" \
    --output-format gif \
    \
	--recep-field-len 5 \
	--batch-size 128 \
	--num-layers 9 \
	--embed-dim-gpt 336 \
	--nb-code 512 \
	--n-head-gpt 8 \
	--block-size 51 \
	--ff-rate 2 \
	--drop-out-rate 0.1 \
	--vq-name PMVQVAE \
	--out_dir output \
	--total-iter 300000 \
	--lr-scheduler 20000 \
	--lr 0.0001 \
	--gamma 0.1 \
	--dataname t2m \
	--down-t 2 \
	--depth 3 \
	--quantizer ema_reset \
	--eval-iter 1000 \
	--pkeep 0.5 \
	--dilation-growth-rate 3 \
	--vq-act relu \
	--vq-resume-setting-pth output/PMVQVAE/vq_setting.json \
	--lptgpt-resume-setting-pth output/LPTGPT/lpt_gpt_setting.json
```


To produce generation results in BVH extension, run the following command.
```
python -m tools.render_animate \
    --seed 122 \
    --texts "a person walks forward and gets pushed, he stumbles to the left and then returns to his original path" \
    --render-form bvh \
    --with-floor \
    --output-dir render_bvh \
    --output-fn "pushed" \
    --output-format bvh \
    \
	--recep-field-len 5 \
	--batch-size 128 \
	--num-layers 9 \
	--embed-dim-gpt 336 \
	--nb-code 512 \
	--n-head-gpt 8 \
	--block-size 51 \
	--ff-rate 2 \
	--drop-out-rate 0.1 \
	--vq-name PMVQVAE \
	--out_dir output \
	--total-iter 300000 \
	--lr-scheduler 20000 \
	--lr 0.0001 \
	--gamma 0.1 \
	--dataname t2m \
	--down-t 2 \
	--depth 3 \
	--quantizer ema_reset \
	--eval-iter 1000 \
	--pkeep 0.5 \
	--dilation-growth-rate 3 \
	--vq-act relu \
	--vq-resume-setting-pth output/PMVQVAE/vq_setting.json \
	--lptgpt-resume-setting-pth output/LPTGPT/lpt_gpt_setting.json
```

The produced BVH can be retargeting to your chosen agents using a Blender Addon [ReTargeting Addon](https://github.com/jack111331/Keemap-Blender-Rig-ReTargeting-Addon)