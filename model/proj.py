from __future__ import print_function

import torch
from torch import nn 
import os
from model.base import BaseBuilder
import clip
from einops import rearrange
from dataloader.classNamesAndTemplates import imagenet_templates
from dataloader.load_descriptions import load_gpt_descriptions
import torch.nn.functional as F
from model.model_utils import get_classembedding, get_desembedding
from model.SMKD import vit_small

        
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


class ProjBuilder(BaseBuilder):
    def __init__(self, opt):
        super().__init__(opt)      
        
        self.enc_t, hdim_t = get_imageEnc(opt.clip_backbone) # attnpool is removed 
        self.enc_s, hdim_s = get_imageEnc(opt.backbone) # cls head is removed
        whole_weight_path = opt.test_weight
        checkpoint = torch.load(
            whole_weight_path, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        filtered_state_dict = {key: value for key, value in state_dict.items() if "enc_s" in key}
        for key in list(filtered_state_dict.keys()):
                filtered_state_dict[key[6:]] = filtered_state_dict.pop(key)
        self.enc_s.load_state_dict(filtered_state_dict, strict=False) # load few-shot vision encoder
        
        self.proj = nn.Linear(hdim_s, hdim_t)
        prototypes = self.init_prototypes(opt, imagenet_templates)
        self.prototypes = nn.Linear(hdim_t, prototypes.shape[0], bias=False) 
        self.prototypes.weight.data.copy_(prototypes)
        
        self.class2descriptions, self.embedding_dict = self.init_embedding_dict(opt, imagenet_templates)
    
    def init_prototypes(self, opt, templates):
        gpt_descriptions, unmodify_dict = load_gpt_descriptions(opt.load_json_for_train_proj, mode=None)
        prototypes = []
        with torch.no_grad():
            for classname, description in gpt_descriptions.items():
                class_embedding = get_classembedding(self.enc_t, classname, templates)
                
                prototypes.append(torch.cat([class_embedding], dim=0)) 
                
            prototypes = torch.cat(prototypes, dim=0)
        return prototypes
    
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
    
    def forward(self, img, selected_classes:list =None, use_ext_data=False, split='train'):
        with torch.no_grad():
            if use_ext_data:
                feat_s, feat_t = [i.float() for i in img]
                if len(feat_t.shape) == 4:
                    feat_t = self.enc_t.encode_image(feat_t).float() 
            else: 
                feat_s, featmap_s = self.enc_s(img)
                feat_t = self.enc_t.encode_image(img).float()
        emb_dim = feat_s.size(-1)
        class_names = [tup[0] for tup in selected_classes]
        # class emb
        proto_sem = torch.cat([self.embedding_dict[class_name][0:1] for class_name in class_names], dim=0).unsqueeze(0).to(feat_s.dtype)
   
        
        feat_p = self.proj(feat_s)
        emb_dim_p = feat_p.size(-1)
        
        if split in ["test", "val"]:
            n_ways = self.opt.n_ways
            n_queries = self.opt.n_queries
            n_shots = self.opt.n_shots
        
        else:
            n_ways = self.opt.n_train_ways
            n_queries = self.opt.n_train_queries
            n_shots = self.opt.n_train_shots
        
        support_idx, query_idx = self.split_instances(n_ways, n_queries, n_shots)
    
        # zero-shot learning
        query_p = feat_p[query_idx.flatten()].view(*(query_idx.shape + (-1,)))
        
        query_p_ = query_p / query_p.norm(dim=-1, keepdim=True)
        proto_sem_ = proto_sem / proto_sem.norm(dim=-1, keepdim=True)
        logits_sem = self.compute_cosLogits(proto_sem_, query_p_, emb_dim_p) # zero-shot inference

        if not self.training:
            return logits_sem
        
        # vision inference
        query = feat_s[query_idx.flatten()].view(*(query_idx.shape + (-1,)))
        query_ = query / query.norm(dim=-1, keepdim=True)
        
        support = feat_s[support_idx.flatten()].view(
            *(support_idx.shape + (-1,))) # b, k, n ,emb_dim
        proto = support.mean(dim=1)  # b, k, n, dim
        proto_ = proto / proto.norm(dim=-1, keepdim=True)
        logits_vis = self.compute_cosLogits(proto_, query_, emb_dim) # 基础Few-shot inference
        
        # mapping l2 loss
        l2 = F.mse_loss(feat_p, feat_t)
        
        # swav: similarity with cluster centers
        sim = self.prototypes(F.normalize(torch.cat((feat_p, feat_t), dim=0), dim=1, p=2)) #
    
        return logits_vis, logits_sem, l2, sim
        
    def freeze_modules(self,):
        for param in self.enc_t.parameters():
            param.requires_grad = False
        for param in self.enc_s.parameters():
            param.requires_grad = False
        for param in self.prototypes.parameters():
            param.requires_grad = False
        
