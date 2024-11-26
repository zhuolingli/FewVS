import torch.nn.functional as F
import torch
import clip

def get_classembedding(clip_model, text, templates):
    text = text.replace('_', ' ')
    texts = [t.format(text) for t in templates]
    texts = clip.tokenize(texts).cuda()
    # prompt ensemble for ImageNet
    class_embeddings = clip_model.encode_text(texts) 
    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    class_embedding = class_embeddings.mean(dim=0, keepdim=True)
    class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
    return class_embedding

def get_desembedding(clip_model, text, templates):
    text = text.replace('_', ' ')
    name, des = text.split(', which')
    
    texts = [t.format(name)[:-1] + ',' + ' which' +  des + '.'  for t in templates]
    texts = clip.tokenize(texts).cuda()
    # prompt ensemble for ImageNet
    class_embeddings = clip_model.encode_text(texts) 
    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    class_embedding = class_embeddings.mean(dim=0, keepdim=True)
    class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
    return class_embedding


def wordlist2string(wordlist):
    string = ', '.join(wordlist)
    
    return string
