import os
import openai
import json

import itertools
import pickle as pkl
from descriptor_strings import stringtolist

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

openai.api_key = '' # GPT3.5 API 
# openai.api_base = ''
os.environ['GPT_ENGINE'] = 'gpt-3.5-turbo-instruct'

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

def generate_prompt(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are visual details for distinguishing a house finch in a photo?
A: There are several useful visual details of a house finch in a photo:
- small bird
- red, brown, or grey
- conical beak
- notched tail
- a short, forked tail
- a brown or grey back
- chest with streaks or spots
- white underbelly
- sometimes a red or orange head and chest in males
- typically found perching or feeding in urban or suburban area

Q: What are visual details for distinguishing a jellyfish in a photo?
A: There are several useful visual details of a jellyfish in a photo:
- umbrella-like shape
- translucent or transparent body
- usually lacks solid structure
- tentacles with stinging cells underneath
- sometimes bright colors
- movement by pulsating motion
- spherical shape
- white with colored panels
- typically 18 panels with 6 equal sections of three panels
- texture for gri

Q: What are visual details for distinguishing a {category_name} in a photo?
A: There are several useful visual details of a {category_name} in a photo:
"""

# generator 
def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))
        
def obtain_descriptors(class_list):
    responses = {}
    descriptors = {}
    
    prompts = [generate_prompt(category.replace('_', ' ')) for category in class_list]
    
    # most efficient way is to partition all prompts into the max size that can be concurrently queried from the OpenAI API
    responses = [completion_with_backoff(
                                        model=os.environ['GPT_ENGINE'],
                                        prompt=prompt_partition,
                                        temperature=0.,
                                        max_tokens=100,
                                            ) for prompt_partition in partition(prompts, 20)]
    response_texts = [r["text"] for resp in responses for r in resp['choices']]
    descriptors_list = [stringtolist(response_text) for response_text in response_texts]
    descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    return descriptors
    
def get_all_classnames(data_root, save_root, dataset):
        
        classnames = {}
        
        for split in ['train', 'val', 'test']:
            
            
            
            pkl_path = os.path.join(data_root, dataset, split + '_label2class.pkl')
            with open(pkl_path, 'rb') as f:
                split_label2class = pkl.load(f)

            split_classnames = [v[0] for i, v in split_label2class.items()]
            classnames[split] = split_classnames
        
        save_path = os.path.join(save_root, dataset +'.json')

        with open(save_path, 'w') as fp:
            json.dump(classnames, fp)


if __name__ == '__main__':
    data_root = 'dataloader/label2class'
    
    save_root = 'GPT3/descriptors'
    # dataset = 'miniImageNet'
    # dataset = 'tieredImageNet'
    # dataset = 'FC100'
    dataset = 'CIFAR_FS'
    
    all_classnames_json_file = os.path.join(save_root, dataset + '.json')
    if not os.path.exists(all_classnames_json_file):
        print('all_classnames_json_file does not exist!')
        get_all_classnames(data_root, save_root, dataset)
        
    all_classnames = json.load(open(all_classnames_json_file, 'r'))
    all_descriptors = {}
    if type(all_classnames) == list:
        all_descriptors.update(obtain_descriptors(all_classnames))
    else:
        for split in ([ 'train', 'val', 'test']):
            split_classnames = all_classnames[split]
            split_descriptors = obtain_descriptors(split_classnames)
            split_descriptors = split_descriptors
            with open(os.path.join(save_root, 'descriptors_' + dataset + '_' + split + '.json'), 'w') as fp:
                json.dump(split_descriptors, fp)
            all_descriptors.update(split_descriptors)
    with open(os.path.join(save_root, 'descriptors_' + dataset + '.json'), 'w') as fp:
        json.dump(all_descriptors, fp)
    
    pass
        
    
    