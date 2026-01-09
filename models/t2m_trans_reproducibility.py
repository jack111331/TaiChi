import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding_reproducibility as pos_encoding
import os
import json
import clip

# Code adopt from VQ-Transformer
# https://github.com/transformer-vq/transformer_vq/blob/main/src/transformer_vq/nn/prob.py
def nucleus(logits, p):
    n_vocab = logits.shape[-1]
    # compute probabilities
    probs = F.softmax(logits, dim=-1)
    # sort probabilities in ascending order and get their argsort indices.
    sorted_probs, sort_indices = torch.sort(probs, dim=-1)
    cumulative_sorted_probs = torch.cumsum(sorted_probs, dim=-1)
    # create a nucleus mask for the sorted probabilities.
    # the mask accepts the largest probabilities whose sum is less than or equal to p,
    # and always includes the largest probability token.
    m1 = torch.greater(cumulative_sorted_probs, 1.0 - (p - 1e-4))  # "is tail > 1-p"?
    m2 = torch.eq(
        torch.arange(n_vocab),
        torch.full(size=[n_vocab], fill_value=n_vocab - 1)
    ).to(logits.device)
    mask_for_sorted = torch.logical_or(m1, m2)
    # unsort the mask so that it applies to the token logits in their non-sorted order.
    unsort_indices = torch.argsort(sort_indices, dim=-1)
    mask = torch.take_along_dim(mask_for_sorted, unsort_indices, dim=-1)
    # mask out the non-nucleus logits.
    masked_logits = logits.masked_fill(~mask, float('-inf'))
    return masked_logits

# Borrowed from https://github.com/exitudio/MMM/blob/main/train_t2m_trans.py#L69 and https://github.com/openai/CLIP/blob/main/clip/model.py#L343
class CLIPOutputWrapper(torch.nn.Module):
    def __init__(self, clip_model) :
        super(CLIPOutputWrapper, self).__init__()
        self.clip_model = clip_model
        
    def forward(self, text_prompts):
        with torch.no_grad():
            text_tokens = clip.tokenize(text_prompts, truncate=True).cuda()

            word_emb = self.clip_model.token_embedding(text_tokens).type(self.clip_model.dtype) # [batch_size, n_ctx, d_model]
            word_emb = word_emb + self.clip_model.positional_embedding.type(self.clip_model.dtype)
            word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
            word_emb = self.clip_model.transformer(word_emb)
            word_feature = self.clip_model.ln_final(word_emb).permute(1, 0, 2).float()
            sent_feature = self.clip_model.encode_text(text_tokens).float()

            # Calculate word_emb length using text
            bs = sent_feature.shape[0]
            padded_text = torch.cat([text_tokens, torch.zeros((bs, 1), dtype=torch.int32, device=text_tokens.device)], dim=1)
            text_srt_idx = torch.arange(padded_text.shape[1], 0, -1, device=padded_text.device).unsqueeze(0)
            text_ref_idx = (padded_text == 0) * text_srt_idx # 0 is the clip's padding
            word_length = torch.argmax(text_ref_idx, dim=1)
            
        return sent_feature, word_feature, word_length

class Text2Motion_Transformer_Word_CrossAtt(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                n_global_head=4,
                n_local_head=2,
                n_mix_head=2,
                drop_out_rate=0.1, 
                fc_rate=4,
                cond_drop_prob=0.1,
                recep_field_len=4):
        super().__init__()
        self.trans_base = CrossCondTransBase_Word_CrossAtt(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, n_global_head, n_local_head, n_mix_head, drop_out_rate, fc_rate, recep_field_len=recep_field_len)
        self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, n_global_head, n_local_head, n_mix_head, drop_out_rate, fc_rate, recep_field_len=recep_field_len)
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_vq = num_vq
        self.end_idx = num_vq
        self.pad_idx = num_vq + 1
        self.cond_drop_prob = cond_drop_prob
                
        self.learnable_temperature = torch.nn.Parameter(torch.Tensor([10.0]))
                
    @staticmethod
    def load_from_setting(setting_path, ckpt_type):
        with open(setting_path, 'r') as f:
            lpt_gpt_settings = json.load(f)
            
        # Complement relpath to abspath
        if ckpt_type in lpt_gpt_settings['checkpoint_dict']:
            ckpt_path = os.path.join(os.path.dirname(setting_path), lpt_gpt_settings['checkpoint_dir'], lpt_gpt_settings["checkpoint_dict"][ckpt_type])
        else:
            raise ValueError(f"LPT-GPT: Checkpoint type {ckpt_type} not found.")

        lpt_gpt_settings.pop("checkpoint_dir", None)
        lpt_gpt_settings.pop("checkpoint_dict", None)
        if os.path.exists(ckpt_path):
            print ('loading transformer checkpoint from {}'.format(ckpt_path))
            lpt_gpt = Text2Motion_Transformer_Word_CrossAtt(**lpt_gpt_settings)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            lpt_gpt.load_state_dict(ckpt['trans'], strict=True)
            
        else:
            raise RuntimeError(f"LPT-GPT checkpoint {ckpt_path} not found")
            
        return lpt_gpt


    def _mask_cond(self, cond, use_cfg=False, use_mask=False):
        # cond (clip_feature, word_emb)
        clip_feature, word_emb = cond
        b, _ = clip_feature.shape

        # Synthesize condition mask where mask=0 means no condition
        if use_cfg:
            mask = torch.cat([torch.ones((b)), torch.zeros((b))], dim=0)
            clip_feature = torch.cat([clip_feature, clip_feature], dim=0)
            word_emb = torch.cat([word_emb, word_emb], dim=0)
        elif use_mask:
            mask = torch.zeros((b))
        elif self.training and self.cond_drop_prob > 0.:
            mask = 1 - torch.bernoulli(self.cond_drop_prob * torch.ones((b)))
        else:
            mask = torch.ones((b))

        mask = mask.type_as(clip_feature)
        return (clip_feature * mask.view(-1, 1), word_emb * mask.view(-1, 1, 1), mask)

    def forward(self, 
                idxs, 
                clip_feature, 
                upper_or_lower_mask, 
                word_emb, 
                word_length,
                kv_cache=None,
                use_cfg=False,
                use_mask=False,
                output_attention=False):
        idxs_up, idxs_down = idxs
        if kv_cache is not None:
            base_kv_cache, head_kv_cache = kv_cache[:self.num_layers], kv_cache[self.num_layers:]
        else:
            base_kv_cache, head_kv_cache = None, None
        
        clip_feature, word_emb, cond_mask = self._mask_cond((clip_feature, word_emb), use_cfg=use_cfg, use_mask=use_mask)
        if use_cfg:
            idxs_up = torch.cat([idxs_up, idxs_up], dim=0)
            idxs_down = torch.cat([idxs_down, idxs_down], dim=0)
            upper_or_lower_mask = torch.cat([upper_or_lower_mask, upper_or_lower_mask], dim=0)
            word_length = torch.cat([word_length, word_length], dim=0)
            
        trans_base_outputs = self.trans_base(idxs_up, idxs_down, clip_feature, upper_or_lower_mask, word_emb, word_length, cond_mask, kv_cache=base_kv_cache, output_attention=output_attention) # attended features, kv_cache, (attention)
        trans_base_output = trans_base_outputs[0]

        trans_head_outputs = self.trans_head(trans_base_output, cond_mask, kv_cache=head_kv_cache, cur_tokenlen=(2 + idxs_up.shape[1] + idxs_down.shape[1]), output_attention=output_attention) # upper body logits, lower body logits, kv_cache, (attention)

        outputs = trans_head_outputs[:2]
        if output_attention:
            base_attns = trans_base_outputs[1]
            head_attns = trans_head_outputs[2]
            attn_dicts = {**base_attns, **head_attns}
            outputs += (attn_dicts, )

        return outputs # upper body logits, lower body logits, (attention dictionary)

    @torch.no_grad()
    def sample_fast_nucleus(self, clip_feature, word_emb, word_length, cfg_scale=3, if_categorial=False, nucleus_p=0.85, use_kv_cache=False):
        device = clip_feature.device
        b = clip_feature.shape[0]
        sample_max_len = self.block_size - 1
        xs_up = torch.full((b, sample_max_len), self.pad_idx, dtype=torch.long, device=device)
        xs_down = torch.full((b, sample_max_len), self.pad_idx, dtype=torch.long, device=device)
        predict_token = torch.zeros((b, 1), dtype=torch.long, device=device) # 0 stands for predict upper, while 1 stands for predict lower, 1-predict_token will invert the predict token
        
        if use_kv_cache:
            upper_kv_cache, lower_kv_cache = [], []
            for _ in range(self.num_layers * 2):
                upper_kv_cache.append({"key": torch.zeros((2*b, self.n_head, (self.block_size+2) * 2, self.embed_dim // self.n_head)).to(device), "value": torch.zeros((2*b, self.n_head, (self.block_size+2) * 2, self.embed_dim // self.n_head)).to(device), "key_mix_local": torch.zeros((2*b, 2, (self.block_size+2) * 2, self.embed_dim // self.n_head)).to(device)})
                lower_kv_cache.append({"key": torch.zeros((2*b, self.n_head, (self.block_size+2) * 2, self.embed_dim // self.n_head)).to(device), "value": torch.zeros((2*b, self.n_head, (self.block_size+2) * 2, self.embed_dim // self.n_head)).to(device), "key_mix_local": torch.zeros((2*b, 2, (self.block_size+2) * 2, self.embed_dim // self.n_head)).to(device)})
        else:
            upper_kv_cache, lower_kv_cache = None, None
        
        for cur_pos in range(sample_max_len):
            # Sample upper-body motion token
            # self.forward implicitly duplicate unconditional batches to accelerate CFG computation
            # outputs = self.forward((xs_up[:, :cur_pos], xs_down[:, :cur_pos]), clip_feature, predict_token, word_emb, word_length, kv_cache=upper_kv_cache, use_cfg=True)
            outputs = self.forward((xs_up[:, :cur_pos], xs_down[:, :cur_pos]), clip_feature, predict_token, word_emb, word_length, kv_cache=upper_kv_cache)
            aux_outputs = self.forward((xs_up[:, :cur_pos], xs_down[:, :cur_pos]), clip_feature, predict_token, word_emb, word_length, kv_cache=upper_kv_cache, use_mask=True)

            logits_up, _ = outputs
            aux_logits_up, _ = aux_outputs
            logits_up = logits_up[:, -1, :]
            aux_logits_up = aux_logits_up[:, -1, :]
            logits_up = aux_logits_up + (logits_up - aux_logits_up) * cfg_scale 
            filtered_logits_up = nucleus(logits_up, p=nucleus_p)
            probs_up = F.softmax(filtered_logits_up, dim=-1)
            if if_categorial:
                dist_up = Categorical(probs_up)
                sampled_idx_up = dist_up.sample()
            else:
                _, sampled_idx_up = torch.topk(probs_up, k=1, dim=-1)
                sampled_idx_up = sampled_idx_up.squeeze(-1)

            xs_up[:, cur_pos] = sampled_idx_up

            # outputs = self.forward((xs_up[:, :cur_pos+1], xs_down[:, :cur_pos]), clip_feature, 1-predict_token, word_emb, word_length, kv_cache=lower_kv_cache, use_cfg=True)
            outputs = self.forward((xs_up[:, :cur_pos+1], xs_down[:, :cur_pos]), clip_feature, 1-predict_token, word_emb, word_length, kv_cache=lower_kv_cache)
            aux_outputs = self.forward((xs_up[:, :cur_pos+1], xs_down[:, :cur_pos]), clip_feature, 1-predict_token, word_emb, word_length, kv_cache=lower_kv_cache, use_mask=True)

            _, logits_down = outputs
            _, aux_logits_down = aux_outputs
            logits_down = logits_down[:, -1, :]
            aux_logits_down = aux_logits_down[:, -1, :]
            logits_down = aux_logits_down + (logits_down - aux_logits_down) * cfg_scale 
            filtered_logits_down = nucleus(logits_down, p=nucleus_p)
            probs_down = F.softmax(filtered_logits_down, dim=-1)
            if if_categorial:
                dist_down = Categorical(probs_down)
                sampled_idx_down = dist_down.sample()
            else:
                _, sampled_idx_down = torch.topk(probs_down, k=1, dim=-1)
                sampled_idx_down = sampled_idx_down.squeeze(-1)

            # Prevent from pad_idx behind end_idx being modified
            xs_down[:, cur_pos] = sampled_idx_down
            
        # To discriminate from full predicted indices and non indices situation, we pad additional end_idx
        xs_up = torch.cat([xs_up, torch.full((b, 1), self.end_idx, device=xs_up.device, dtype=torch.long)], dim=1)
        # https://stackoverflow.com/questions/56088189/pytorch-how-can-i-find-indices-of-first-nonzero-element-in-each-row-of-a-2d-ten
        srt_idx = torch.arange(xs_up.shape[1], 0, -1, device=xs_up.device).unsqueeze(0)
        ref_idx = (xs_up == self.end_idx) * srt_idx # [[0, 0, 0, 1, 0], ...] * [[0., -0.2, -0.4, -0.6, -0.8]] -> [[0, 0, 0, -0.6, 0], ...]
        pred_token_len = torch.argmax(ref_idx, dim=1)

        return xs_up, xs_down, pred_token_len

class CausalCrossConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, n_global_head=4, n_local_head=2, n_mix_head=2, drop_out_rate=0.1, recep_field_len=4):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.block_size = block_size
        self.n_head = n_head
        self.n_local_head = n_local_head
        self.n_global_head = n_global_head
        self.n_mix_head = n_mix_head

        self.key_mix_local = nn.Linear(embed_dim, embed_dim // n_head * self.n_mix_head)
        self.query_mix_local = nn.Linear(embed_dim, embed_dim // n_head * self.n_mix_head)
        self.mix_conv = nn.Conv2d(self.n_mix_head * 2, self.n_mix_head, groups=self.n_mix_head, kernel_size=1, bias=False)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)

        # Compute global head's mask
        temporal_orders = torch.cat([torch.zeros((2)), torch.arange(block_size) * 2 + 1, torch.arange(block_size) * 2 + 2], dim=0)
        seq_mask = temporal_orders.view(1, block_size * 2 + 2) <= temporal_orders.view(block_size * 2 + 2, 1)
        global_mask = torch.zeros((block_size * 2 + 2, block_size * 2 + 2)).masked_fill(seq_mask, 1).repeat(1, self.n_mix_head + self.n_global_head, 1, 1)

        # Compute local head's mask
        temporal_orders_local_query = torch.cat([torch.arange(block_size) * 2 + 1 - 2 * recep_field_len + 1, torch.arange(block_size) * 2 + 1 - 2 * recep_field_len + 1], dim=0)
        temporal_orders_local_key = torch.cat([torch.arange(block_size) * 2 + 1, torch.arange(block_size) * 2 + 1], dim=0)
        de_seq_mask = temporal_orders_local_key.view(1, block_size * 2) < temporal_orders_local_query.view(block_size * 2, 1)
        local_seq_mask = seq_mask.clone()
        local_seq_mask[2:, 2:] = local_seq_mask[2:, 2:] & (~de_seq_mask)
        local_mask = torch.zeros((block_size * 2 + 2, block_size * 2 + 2)).masked_fill(local_seq_mask, 1).repeat(1, self.n_local_head, 1, 1)

        self.register_buffer("mask", torch.cat([local_mask, global_mask], dim=1))
        
        # Declare learnable attention bias
        self.att_bias_clip = nn.Parameter(torch.ones((1, n_head, 1), requires_grad=True) * 0.5) # 1 Batch, head, 1 key
        self.att_bias_clip_mix_local = nn.Parameter(torch.ones((1, self.n_mix_head, 1), requires_grad=True) * 0.5) # 1 Batch, head, 1 key

    def forward(self, x, cond_mask, kv_cache=None, cur_tokenlen=50, output_attention=False):
        # 5/23 convert to seq causal attention
        '''
        In inference, the "predict task token (upper-body or lower-body)" and CLIP sentence-level token should be simultaneously visible
        0p: predict task token, 0c: CLIP sentence-level token, 1u: t=1 upper-body token, 1d: t=1 lower-body token, rest tokens etc. 
        Global mask:
            0p   0c   1u   1d   2u   2d                               0p   0c   1u   2u   1d   2d
        0p   1    1                                               0p   1    1
        0c   1    1                                               0c   1    1
        1u   1    1    1                      represented as ->   1u   1    1    1       
        1d   1    1    1    1                                     2u   1    1    1    1    1    
        2u   1    1    1    1    1                                1d   1    1    1         1 
        2d   1    1    1    1    1    1                           2d   1    1    1    1    1    1

        We define the future key motion token as the token which is non-visible to the current query token, including 
        temporal order (t=2 upper/lower-body token is not visible to t=1 upper/lower-body token) and body-part order (t=2 lower-body key token is not visible to t=2 upper-body query token, because we defined
        our synthesis order to be first upper-body token and then lower-body token)
        We treat a full-body token (containing an upper-body and a lower-body tokens "both in the same temporal position", if the lower-body token is the future key token, then it is not present in the full-body token)
        as the minimal token unit, we mask out the full-body token outside the receptive field. 
        Local mask (Sliding window = 1)
            0p   0c   1u   1d   2u   2d                               0p   0c   1u   2u   1d   2d
        0p   1    1                                               0p   1    1
        0c   1    1                                               0c   1    1
        1u   1    1    1                      represented as ->   1u   1    1    1     
        1d   1    1    1    1                                     2u   1    1         1    
        2u   1    1              1                                1d   1    1    1         1
        2d   1    1              1    1                           2d   1    1         1         1

        Local mask (Sliding window = 2)
            0p   0c   1u   1d   2u   2d   3u   3d                             0p   0c   1u   2u   3u   1d   2d   3d
        0p   1    1                                                       0p   1    1                    
        0c   1    1                                                       0c   1    1                    
        1u   1    1    1                              represented as ->   1u   1    1    1                
        1d   1    1    1    1                                             2u   1    1    1    1         1                 
        2u   1    1    1    1    1                                        3u   1    1         1    1         1    
        2d   1    1    1    1    1    1                                   1d   1    1    1              1
        3u   1    1              1    1    1                              2d   1    1    1    1         1    1
        3d   1    1              1    1    1    1                         3d   1    1         1    1         1    1

        
        current frame upper can't see current frame lower, but current frame lower can see current frame upper
        The sliding window=1 represent it can only see current frame's information, and because of sequential generation, upper can only see self upper token, while lower can see self upper and lower token
        The sliding window=t represent it can see current frame and previous t-1 frames information
        The previous {n_head * global_local_ratio} heads are local heads, while next {n_head - (n_head * global_local_ratio)} heads are global heads
        '''
        B, T, C = x.size() # T = 2 + upper_body_t (upper-body tokens) + lower_body_t (lower-body tokens)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if self.training == False and kv_cache is not None:
            k = kv_cache['key']
            v = kv_cache['value']
            k_mix_local = kv_cache['key_mix_local']

            # [B, H, T, C]
            upper_body_t = (cur_tokenlen-1)//2
            lower_body_t = cur_tokenlen-2-upper_body_t
            upper_body_t = max(upper_body_t, 0)
            lower_body_t = max(lower_body_t, 0)
            
            # cache_mapping: Calculate new key-value cache mapping to its position in key-value cache, the cache for key and value is layout as [1 (prediction task token), 1 (CLIP sentence-level token), block_size (upper-body tokens), block_size (lower-body tokens)]
            # t_ub_bound_q, lb_bound_q, t_ub_bound_k, lb_bound_k: Calculate the parts to extract attention mask applying on current calculation
            if cur_tokenlen <= 3:
                # cur_tokenlen=2 (predicting 1st upper-body token), cache key-value with position 0 (prediction task token), 1 (CLIP sentence-level token)
                # cur_tokenlen=3 (predicting 1st lower-body token), cache key-value with position 0 (prediction task token), 1 (CLIP sentence-level token), 2 (1st upper-body token)
                cache_mapping = [i for i in range(cur_tokenlen)]
                t_ub_bound_q = [0, cur_tokenlen]
                lb_bound_q = [2+self.block_size, 2+self.block_size]
                t_ub_bound_k = [0, cur_tokenlen]
                lb_bound_k = [2+self.block_size, 2+self.block_size]

            else:
                # cur_tokenlen=4 (predicting 2nd upper-body token), cache key-value with position 2 (1st upper-body token), 2+block_size (1st lower-body token)
                # cur_tokenlen=5 (predicting 2md lower-body token), cache key-value with position 3 (2nd upper-body token), 2+block_size (1st lower-body token)
                cache_mapping = [2 + (upper_body_t-1), 2 + self.block_size + (lower_body_t-1)]
                t_ub_bound_q = [2+(upper_body_t-1), 2+upper_body_t]
                lb_bound_q = [2+self.block_size+(lower_body_t-1), 2+self.block_size+lower_body_t]                
                t_ub_bound_k = [0, 2+upper_body_t]
                lb_bound_k = [2+self.block_size, 2+self.block_size+lower_body_t]

            # Update key-value cache and use current key-value caches
            k[:, :, cache_mapping] = self.key(x).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
            k = torch.cat([k[:, :, t_ub_bound_k[0]:t_ub_bound_k[1]], k[:, :, lb_bound_k[0]:lb_bound_k[1]]], dim=2)
            
            q = self.query(x).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v[:, :, cache_mapping] = self.value(x).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
            v = torch.cat([v[:, :, t_ub_bound_k[0]:t_ub_bound_k[1]], v[:, :, lb_bound_k[0]:lb_bound_k[1]]], dim=2)
            
            k_mix_local[:, :, cache_mapping] = self.key_mix_local(x).view(B, -1, self.n_mix_head, C // self.n_head).transpose(1, 2)
            k_mix_local = torch.cat([k_mix_local[:, :, t_ub_bound_k[0]:t_ub_bound_k[1]], k_mix_local[:, :, lb_bound_k[0]:lb_bound_k[1]]], dim=2)
            q_mix_local = self.query_mix_local(x).view(B, -1, self.n_mix_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        else:
            k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            
            k_mix_local = self.key_mix_local(x).view(B, T, self.n_mix_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q_mix_local = self.query_mix_local(x).view(B, T, self.n_mix_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

            upper_t = (T-1) // 2
            lower_t = T-2-upper_t
            t_ub_bound_q = [0, 2+upper_t]
            lb_bound_q = [2+self.block_size, 2+self.block_size+lower_t]
            t_ub_bound_k = [0, 2+upper_t]
            lb_bound_k = [2+self.block_size, 2+self.block_size+lower_t]

        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # The layout of att is [Local heads (2), Mix heads global attention (2), Global heads (4)]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att_mix_local = (q_mix_local @ k_mix_local.transpose(-2, -1)) * (1.0 / math.sqrt(k_mix_local.size(-1)))

        # Amplify the attention of the CLIP sentence-level condition
        # We use torch.where to prevent applying attention bias when unconditional generation
        att[:, :, :, 1] = torch.where(cond_mask.view(-1, 1, 1) > 0, att[:, :, :, 1] + F.relu(self.att_bias_clip) * 10.0, att[:, :, :, 1])
        att_mix_local[:, :, :, 1] = torch.where(cond_mask.view(-1, 1, 1) > 0, att_mix_local[:, :, :, 1] + F.relu(self.att_bias_clip_mix_local) * 10.0, att_mix_local[:, :, :, 1])

        composite_mask_upper = torch.cat([self.mask[:,:,t_ub_bound_q[0]:t_ub_bound_q[1],t_ub_bound_k[0]:t_ub_bound_k[1]], self.mask[:,:,t_ub_bound_q[0]:t_ub_bound_q[1],lb_bound_k[0]:lb_bound_k[1]]], dim=3)
        composite_mask_lower = torch.cat([self.mask[:,:,lb_bound_q[0]:lb_bound_q[1],t_ub_bound_k[0]:t_ub_bound_k[1]], self.mask[:,:,lb_bound_q[0]:lb_bound_q[1],lb_bound_k[0]:lb_bound_k[1]]], dim=3)
        composite_mask = torch.cat([composite_mask_upper, composite_mask_lower], dim=2)
        att = att.masked_fill(composite_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        # Calculate mix head's local attention part
        att_mix_local = att_mix_local.masked_fill(composite_mask[:, :self.n_mix_head] == 0, float('-inf'))
        att_mix_local = F.softmax(att_mix_local, dim=-1)

        # Pair-wise concatenate mix head's global and local attention, Concatenate local, mix, global heads head-wise
        global_local_head = torch.cat([att[:, self.n_local_head:self.n_local_head+self.n_mix_head].unsqueeze(2), att_mix_local.unsqueeze(2)], dim=2).view(B, self.n_mix_head*2, att.shape[-2], att.shape[-1]) # [bs, heads, 2, T, T] -> [bs, 2*heads, T, T]
        att = torch.cat([att[:, :self.n_local_head], self.mix_conv(global_local_head), att[:, self.n_local_head+self.n_mix_head:]], dim=1)

        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, -1, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        outputs = (y, )
        if output_attention:
            outputs += (att, )
        
        return outputs

class CrossAttention(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head

    def forward(self, x, word_emb, non_pad_mask, output_attention=False):
        B, T, C = x.size()
        B, N, D = word_emb.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(word_emb).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(word_emb).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, N) -> (B, nh, T, N)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(non_pad_mask.unsqueeze(1) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, N) x (B, nh, N, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        outputs = (y, )
        if output_attention:
            outputs += (att, )

        return outputs

class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, n_global_head=4, n_local_head=2, n_mix_head=2, drop_out_rate=0.1, fc_rate=4, recep_field_len=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalCrossConditionalSelfAttention(embed_dim, block_size, n_head, n_global_head, n_local_head, n_mix_head, drop_out_rate, recep_field_len=recep_field_len)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, cond_mask, kv_cache=None, cur_tokenlen=50, output_attention=False):
        # 2024/1/24 Block contain attention and mlp, if we want to divide the clip_feature from full-body into upper and lower part, we can only modify attention mechanism,
        # Because MLP sees only the last dimension (embedding)
        attn_outputs = self.attn(self.ln1(x), cond_mask, kv_cache=kv_cache, cur_tokenlen=cur_tokenlen, output_attention=output_attention)
        residual_x = attn_outputs[0]
        x = x + residual_x
        x = x + self.mlp(self.ln2(x))
        output = (x, )
        if output_attention:
            output += (attn_outputs[1], )
        return output

class Block_crossatt(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.attn = CrossAttention(embed_dim, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, word_emb, mask, output_attention=False):
        attn_outputs = self.attn(self.ln1(x), self.ln3(word_emb), non_pad_mask=mask, output_attention=output_attention)
        residual_x = attn_outputs[0]
        x = x + residual_x
        x = x + self.mlp(self.ln2(x))
        outputs = (x, )
        if output_attention:
            outputs += (attn_outputs[1], )

        return outputs

class CrossCondTransBase_Word_CrossAtt(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                n_global_head=4,
                n_local_head=2,
                n_mix_head=2,
                drop_out_rate=0.1, 
                fc_rate=4,
                num_layers_cross=2,
                recep_field_len=5):
        super().__init__()
        self.tok_emb_up = nn.Embedding(num_vq + 2, embed_dim)
        self.tok_emb_down = nn.Embedding(num_vq + 2, embed_dim) # actually, only use num_vq and padding
        self.up_token = nn.Parameter(torch.randn((1, 1, embed_dim), requires_grad=True))
        self.down_token = nn.Parameter(torch.randn((1, 1, embed_dim), requires_grad=True))
        self.pred_tok_emb = nn.Embedding(2, embed_dim)
        self.word_emb = nn.Linear(clip_dim, embed_dim)
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        self.num_vq = num_vq
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size + 1, n_head, n_global_head, n_local_head, n_mix_head, drop_out_rate, fc_rate, recep_field_len=recep_field_len) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size + 1, embed_dim, 0.0, False)
        self.cross_att_upper = nn.ModuleList()
        self.cross_att_lower = nn.ModuleList()
        for i in range(num_layers_cross):
            self.cross_att_upper.append(
                Block_crossatt(embed_dim, n_head, drop_out_rate, fc_rate)
            )
            self.cross_att_lower.append(
                Block_crossatt(embed_dim, n_head, drop_out_rate, fc_rate)
            )

        self.block_size = block_size

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def create_word_crossatt_mask(self, x, word_length):
        b, T = x.shape[:2]
        token_mask = torch.ones((b, T), dtype=torch.bool) # [b, T]
        word_mask = torch.arange(77).expand(b, 77) < word_length.unsqueeze(1).cpu() # [b, N]
        mask = token_mask.view(b, T, 1) * word_mask.view(b, 1, 77)
        return mask.long()

    def forward(self,
                idx_up, 
                idx_down, 
                clip_feature, 
                upper_or_lower_mask, 
                word_emb, 
                word_length,
                cond_mask,
                kv_cache=None,
                output_attention=False):
        # TODO kv_cache currently only store self-attention ones
        # upper_or_lower_mask: 0 stands for predict upper, while 1 stands for predict lower
        if idx_up.shape[1] - idx_down.shape[1] >= 1:
            idx_down = torch.cat([idx_down, torch.ones((idx_up.shape[0], 1), dtype=torch.long, device=idx_up.device) * (self.num_vq + 1)], dim=1)
        
        attn_dict = {"cross_attn_up": [], "cross_attn_down": [], "base_attn": []}
        if idx_up.shape[1] == 0 and idx_down.shape[1] == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            b, t = idx_up.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            b, t = idx_down.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            token_embeddings_up = self.tok_emb_up(idx_up)
            token_embeddings_down = self.tok_emb_down(idx_down)
            token_word_mask_upper = self.create_word_crossatt_mask(token_embeddings_up, word_length).to(token_embeddings_up.device)
            token_word_mask_lower = self.create_word_crossatt_mask(token_embeddings_down, word_length).to(token_embeddings_down.device)
            word_emb = self.word_emb(word_emb)
            
            for module in self.cross_att_upper:
                cross_attn_up_outputs = module(token_embeddings_up, word_emb, mask=token_word_mask_upper, output_attention=output_attention)
                token_embeddings_up = cross_attn_up_outputs[0]
                if output_attention:
                    attn_dict["cross_attn_up"].append(cross_attn_up_outputs[1])
            for module in self.cross_att_lower:
                cross_attn_down_outputs = module(token_embeddings_down, word_emb, mask=token_word_mask_lower, output_attention=output_attention)
                token_embeddings_down = cross_attn_down_outputs[0]
                if output_attention:
                    attn_dict["cross_attn_down"].append(cross_attn_down_outputs[1])
                
            token_embeddings_up = token_embeddings_up + self.up_token
            token_embeddings_down = token_embeddings_down + self.down_token
            token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings_up, token_embeddings_down], dim=1)
        
        token_embeddings = torch.cat([self.pred_tok_emb(upper_or_lower_mask), token_embeddings], dim=1)
        # We use the same positional embedding for upper-body and lower-body token indicating their temporal position
        x = self.pos_embed(token_embeddings)
        if x.shape[1] <= 3:
            pass
        elif kv_cache is not None:
            upper_body_t = (x.shape[1] - 1)//2
            x = torch.cat([x[:, 2+(upper_body_t-1):2+upper_body_t], x[:, -1:]], dim=1)
            
        for blk_idx, blk in enumerate(self.blocks):
            cur_kv_cache = kv_cache[blk_idx] if kv_cache is not None else None
            blk_output = blk(x, cond_mask, kv_cache=cur_kv_cache, cur_tokenlen=token_embeddings.shape[1], output_attention=output_attention)
            x = blk_output[0]
            if output_attention:
                attn_dict["base_attn"].append(blk_output[1])

        outputs = (x, )
        if output_attention:
            outputs += (attn_dict, )
            
        return outputs


class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                n_global_head=4,
                n_local_head=2,
                n_mix_head=2,
                drop_out_rate=0.1, 
                fc_rate=4,
                recep_field_len=4):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, n_global_head, n_local_head, n_mix_head, drop_out_rate, fc_rate, recep_field_len=recep_field_len) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, cond_mask, kv_cache=None, cur_tokenlen=50, output_attention=False):
        N, T, C = x.size()
        upper_t = (T - 2) // 2 # T can be odd, so minus one and use division's nature of discarding can effectively find upper body tokens
        attn_dict = {"head_attn": []}
        for blk_idx, blk in enumerate(self.blocks):
            cur_kv_cache = kv_cache[blk_idx] if kv_cache is not None else None
            blk_outputs = blk(x, cond_mask, kv_cache=cur_kv_cache, cur_tokenlen=cur_tokenlen, output_attention=output_attention)
            x = blk_outputs[0]
            if output_attention:
                attn = blk_outputs[1]
                attn_dict["head_attn"].append(attn)

        x = self.ln_f(x)
        logits = self.head(x)
        if not self.training and kv_cache is not None:
            if cur_tokenlen > 3: 
                # There is two situations here: 
                #    1. logits[0] and logits[1] is predicted from the t upper-body token and the t lower-body token,
                #       their corresponding next token is the t lower-body token and the t+1 upper-body token
                #    2. logits[0] and logits[1] is predicted from the t upper-body token and the t-1 lower-body token,
                #       their corresponding next token is the t lower-body token and the t upper-body token
                logits_up, logits_down = logits[:, 1:2], logits[:, 0:1]
            else:
                # logits[1] and logits[2] is predicted from CLIP sentence token and the t=1 upper-body token,
                # their corresponding next token is t=1 upper-body token and t=2 lower-body token
                logits_up, logits_down = logits[:, 1:2], logits[:, 2:3]
        else:
            logits_up, logits_down = torch.cat([logits[:, 1:2], logits[:, upper_t+2:]], dim=1), logits[:, 2:upper_t+2]

        # Remove END token probability for lower-body token
        logits_down = logits_down[..., :-1]
            
        outputs = (logits_up, logits_down)
        if output_attention:
            outputs += (attn_dict, )
        return outputs