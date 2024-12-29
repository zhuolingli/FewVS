from __future__ import print_function

import torch
from torch import nn, Tensor

from model.base import BaseBuilder

from einops import rearrange
from dataloader.classNamesAndTemplates import imagenet_templates
from dataloader.load_descriptions import load_gpt_descriptions
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from util.util import *
from model.model_utils import get_classembedding, get_desembedding
from model.SMKD import vit_small
import clip

def get_imageEnc(name:str, ckpt_pth=None):
    hidden_dim = 640
    if name == 'identity':
        encoder = nn.Identity()
        
    elif 'RN' in name:
        encoder = clip.load(name)[0]
        hidden_dim = encoder.visual.attnpool.c_proj.out_features
        return encoder, hidden_dim
    elif name == 'Res12':
        from backbone.resnet12 import resnet12
        encoder = resnet12()
    elif name == 'SMKD':
        encoder = vit_small()
        hidden_dim = encoder.norm.weight.shape[0]
    else:
        raise ValueError(f"image encoder {name} not recognized")
    
    return encoder, hidden_dim

class FewVSBuilder(BaseBuilder):
    def __init__(self, opt):
        super().__init__(opt)      
        self.enc_t, hdim_t = get_imageEnc(opt.clip_backbone) # attnpool is removed 
        self.enc_s, hdim_s = get_imageEnc(opt.backbone) # cls head is removed
        self.sem, self.alpha = opt.sem, opt.alpha 
        self.proj = nn.Linear(hdim_s, hdim_t)
        self.fsl_mod_inductive = PatchFSL(self.opt, 1, 1, self.proj)
        # total_params = sum(p.numel() for p in model.parameters())
        self.class2descriptions, self.embedding_dict = self.init_embedding_dict(opt, imagenet_templates)
    
    def init_embedding_dict(self, opt, templates):
        class2descriptions, unmodify_dict = load_gpt_descriptions(opt.dataset, mode='prepend') 
        embedding_dict = {}
        with torch.no_grad():
            for classname, description in class2descriptions.items():
                # get class embedding
                class_embedding = get_classembedding(self.enc_t, classname, templates)
                
                # get description_encodings
                description_embeddings = torch.cat([get_desembedding(self.enc_t, des, templates) for des in description], dim=0) # w special prompt
 
                embedding_dict[classname] = torch.cat([class_embedding, description_embeddings], dim=0)
        
        return class2descriptions, embedding_dict
        
    def run_adapter(self, support, query, support_one_hot, proto_sem, query_p, clip_logits=None):
        # L2 norm
        support, query, query_p = [feat / feat.norm(dim=-1, keepdim=True) for feat in  [support, query, query_p]]        
        support, query, query_p = [feat.reshape(-1, feat.shape[-1]) for feat in  [support, query, query_p]]
        
        # Zero-shot CLIP
        if clip_logits==None:
            clip_logits = 100. * query_p @ proto_sem.squeeze(0).t() # [15,512] * [1000, 512]
        else:
            clip_logits = 1. * clip_logits

        # prototype
        support_num = support_one_hot.sum(0)
        class_sums = support_one_hot.float().t() @ support
        proto = class_sums / support_num[:, None]
        proto_ = proto / proto.norm(dim=-1, keepdim=True)
        query_ = query / query.norm(dim=-1, keepdim=True)
        cache_logits = self.compute_cosLogits(proto_.unsqueeze(0), query_.unsqueeze(0), proto_.shape[-1])
        
        
        logits =  self.sem * clip_logits + cache_logits * self.alpha
        return logits
    
    def instance_infer(self, class_names, feat_s, n_ways, n_queries, n_shots):
        support_idx, query_idx = self.split_instances(n_ways, n_queries, n_shots)
        
        class_emb = torch.cat([self.embedding_dict[class_name][0:1] for class_name in class_names], dim=0).unsqueeze(0).to(feat_s.dtype)
        # get des embeddings
        des_emb_mean = torch.cat([self.embedding_dict[class_name][1:].mean(0).unsqueeze(0) for class_name in class_names], dim=0).unsqueeze(0).to(feat_s.dtype)
        
        # get supp and query
        support_one_hot = F.one_hot(torch.arange(n_ways).repeat(n_shots)).cuda() # k*n, 1
        support = feat_s[support_idx.flatten()].view(*(support_idx.shape + (-1,))) # b, k, n ,emb_dim
        query = feat_s[query_idx.flatten()].view(*(query_idx.shape + (-1,)))
        query_p = self.proj(query)
        
        logits = self.run_adapter(support, query, support_one_hot, des_emb_mean, query_p)
        return logits

    def online_otpim_infer(self, class_names, feat_s, n_ways, n_queries, n_shots):
        if len(feat_s.shape) == 2:
            feat_s = feat_s.unsqueeze(-1).unsqueeze(-1)
        feats_tokens = feat_s.flatten(2).transpose(1,2) # n*k, l, n_dim
        
        des_label = []
        des_embeddings = []
        for meta_label_id, class_name in enumerate(class_names):
            des_embeddings.append(self.embedding_dict[class_name][1:])
            des_label += [meta_label_id] * (self.embedding_dict[class_name][1:].shape[0])
        des_embeddings = torch.cat(des_embeddings, dim=0).to(feat_s.dtype)
        
        support_idx, query_idx = self.split_instances(n_ways, n_queries, n_shots)
        support = feats_tokens[support_idx.flatten()] # k*n ,emb_dim  
        query = feats_tokens[query_idx.flatten()] # q*n
        
        label_supp = torch.arange(n_ways).repeat(n_shots).cuda() # k*n, 1
        
        logits = self.fsl_mod_inductive(support, query, des_embeddings, label_supp, des_label)
        return logits
    
    def mix_infer(self, class_names, feat_s, n_ways, n_queries, n_shots):
        clip_logits = self.online_otpim_infer(class_names, feat_s, n_ways, n_queries, n_shots)
        
        support_idx, query_idx = self.split_instances(n_ways, n_queries, n_shots)
        
        # get cls embeddings
        des_emb_mean = torch.cat([self.embedding_dict[class_name][0:1] for class_name in class_names], dim=0).unsqueeze(0).to(feat_s.dtype)
        
        # get supp and query
        support_one_hot = F.one_hot(torch.arange(n_ways).repeat(n_shots)).cuda() # k*n, 1
        support = feat_s[support_idx.flatten()].view(*(support_idx.shape + (-1,))) # b, k, n ,emb_dim
        query = feat_s[query_idx.flatten()].view(*(query_idx.shape + (-1,)))
        query_p = self.proj(query)
        
        
        logits = self.run_adapter(support, query, support_one_hot, des_emb_mean, query_p, clip_logits)
        return logits
    
    def forward(self, img, selected_classes:list =None, split='train'):
        if split in ["test", "val"]:
            n_ways = self.opt.n_ways
            n_queries = self.opt.n_queries
            n_shots = self.opt.n_shots
        else:
            n_ways = self.opt.n_train_ways
            n_queries = self.opt.n_train_queries
            n_shots = self.opt.n_train_shots
        
        with torch.no_grad():
            feat_s, featmap_s = self.enc_s(img)
        
        class_names = [tup[0] for tup in selected_classes]
        # logits = self.instance_infer(class_names, feat_s, n_ways, n_queries, n_shots)
        # logits = self.online_otpim_infer(class_names, featmap_s, n_ways, n_queries, n_shots)
        logits = self.mix_infer(class_names, feat_s, n_ways, n_queries, n_shots)
        
        return logits
    
    def freeze_modules(self,):
        for param in self.enc_t.parameters():
            param.requires_grad = False
        for param in self.enc_s.parameters():
            param.requires_grad = False
    
def compute_emb_cosine_similarity(support_emb: torch.Tensor, query_emb: torch.Tensor):
    """Compute the similarity matrix C between support and query embeddings using the cosine similarity.
       We reformulate the cosine sim computation as dot product using matrix mult and transpose, due to:
       cossim = dot(u,v)/(u.norm * v.norm) = dot(u/u.norm, v/v.norm);    u/u.norm = u_norm, v/v.norm = v_norm
       For two matrices (tensors) U and V of shapes [n,b] & [m,b] => torch.mm(U_norm,V_norm.transpose)
       This returns a matrix showing all pairwise cosine similarities of all entries in both matrices, i.e. [n,m]"""

    # Note: support represents the 'reference' embeddings, query represents the ones that need classification;
    #       During adaptation of peiv, support set embeddings will be used for both to optimise the peiv
    support_shape = support_emb.shape # [nk, l ,dim]
    support_emb_vect = support_emb.reshape(support_shape[0] * support_shape[1], -1)  # shape e.g. [4900, 384]
    # Robust version to avoid division by zero
    support_norm = torch.linalg.vector_norm(support_emb_vect, dim=1).unsqueeze(dim=1)
    support_norm = support_emb_vect / torch.max(support_norm, torch.ones_like(support_norm) * 1e-8)

    query_shape = query_emb.shape
    query_emb_vect = query_emb.reshape(query_shape[0] * query_shape[1], -1)  # shape e.g. [14700, 384]
    # Robust version to avoid division by zero
    query_norm = query_emb_vect.norm(dim=1).unsqueeze(dim=1)
    query_norm = query_emb_vect / torch.max(query_norm, torch.ones_like(query_norm) * 1e-8)

    return torch.matmul(support_norm, query_norm.transpose(0, 1))  # shape e.g. [4900, 14700]

class PatchFSL(nn.Module): 
    def __init__(self, args, sup_emb_key_seq_len, sup_emb_query_seq_len, proj):
        super(PatchFSL, self).__init__()
        
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.proj = proj
        
        # Mask to prevent image self-classification during adaptation
        if args.n_shots > 1:  # E.g. for 5-shot scenarios, use 'full' block-diagonal logit matrix to mask entire image
            # self.block_mask = torch.block_diag(*[torch.ones(sup_emb_key_seq_len, sup_emb_query_seq_len) * -100.
                                                #  for _ in range(args.n_ways * args.n_shots)]).cuda()
            self.block_mask = torch.block_diag(*[torch.ones(sup_emb_key_seq_len, sup_emb_query_seq_len)
                                                 for _ in range(args.n_ways * args.n_shots)]).bool().cuda()
            self.block_mask = ~ self.block_mask
        else:  # 1-shot experiments require diagonal in-image masking, since no other samples available
            self.block_mask = torch.ones(sup_emb_key_seq_len * args.n_ways * args.n_shots,
                                         sup_emb_query_seq_len * args.n_ways * args.n_shots).cuda()
            self.block_mask = (self.block_mask - self.block_mask.triu(diagonal=args.block_mask_1shot)
                               - self.block_mask.tril(diagonal=-args.block_mask_1shot)) * -100.
        
        
        self.log_tau_c = torch.tensor([np.log(args.similarity_temp_init)], requires_grad=True, device='cuda')
        self.optimiser_str = args.optimiser_online
        self.opt_steps = args.optim_steps_online
        self.lr_online = args.lr_online
        self.loss_fn = torch.nn.CrossEntropyLoss()
        

    def _predict(self, support_emb, query_emb, des_emb, des_labels, phase='infer'):
        """Perform one forward pass using the provided embeddings as well as the module-internal
        patch embedding importance vector 'peiv'. The phase parameter denotes whether the prediction is intended for
        adapting peiv ('adapt') using the support set, or inference ('infer') on the query set."""
        sup_emb_seq_len = support_emb.shape[1]
        # Compute patch embedding similarity
        C = compute_emb_cosine_similarity(des_emb.unsqueeze(0), self.proj(query_emb)) # D, q*lq
        
        # Add patch embedding importance vector (logits, corresponds to multiplication of probabilities)
        pred = torch.add(C, self.v2.unsqueeze(1))  # using broadcasting # [m1+m2+...+mn, q*lq]
        pred = pred.transpose(0, 1).reshape(query_emb.shape[0], query_emb.shape[1], -1) # q,lq, m1+m2+...+mn
        pred = pred / torch.exp(self.log_tau_c)
        pred2_list = []
        counted_des_num = 0
        for meta_id in range(self.n_ways):
            des_num = des_labels.count(meta_id)
            pred_des = pred[:, :, counted_des_num:counted_des_num+des_num].reshape(pred.shape[0], -1) # num_q, l_q*mi
            pred_des = torch.sum(pred_des, dim=1) / pred_des.shape[1] # num_q, 1
            
            counted_des_num += des_num
            pred2_list.append(pred_des.unsqueeze(1))
        pred = torch.cat(pred2_list, dim=1)
        
        return pred

    @torch.enable_grad()
    def _optimise_peiv(self, support_emb_key, support_emb_query, des_emb, supp_labels, des_labels):
        # Detach, we don't want to compute computational graph w.r.t. model
        support_emb_key = support_emb_key.detach()
        support_emb_query = support_emb_query.detach()
        supp_labels = supp_labels.detach()
        des_emb = des_emb.detach()
        
        params_to_optimise = [self.v1, self.v2]
        # Perform optimisation of patch embedding importance vector v; embeddings should be detached here!
        self.optimiser_online = torch.optim.SGD(params=params_to_optimise, lr=self.lr_online)
        self.optimiser_online.zero_grad()
        # Run for a specific number of steps 'self.opt_steps' using SGD
        for s in range(self.opt_steps):
            support_pred = self._predict(support_emb_key, support_emb_query, des_emb, des_labels, phase='adapt')
            loss = self.loss_fn(support_pred, supp_labels)
            loss.backward()
            self.optimiser_online.step()
            self.optimiser_online.zero_grad()
        # Set initialisation/reset flag to False since peiv is no longer 'just' initialised
        return
    def forward(self, support_emb_key, query_emb, des_embs, support_labels, des_labels):
        
        self.v1 = torch.zeros(self.n_shots * self.n_ways * support_emb_key.shape[1], requires_grad=True, device="cuda")
        self.v2 = torch.zeros(len(des_labels), requires_grad=True, device="cuda")
        
        self._optimise_peiv(support_emb_key, support_emb_key, des_embs, support_labels, des_labels)
        # Retrieve the predictions of query set samples
        pred_query = self._predict(support_emb_key, query_emb, des_embs, des_labels, phase='infer')
        return pred_query
    
    
    