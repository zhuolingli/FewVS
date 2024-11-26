import nltk
from nltk.corpus import wordnet

import os
import openai
import json

import itertools
import pickle as pkl
from tqdm import tqdm

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# os.environ['https_proxy'] = '127.0.0.1:7890' # clash port 
openai.api_key = '' #FILL IN YOUR OWN HERE
# openai.api_base = "" 
os.environ['GPT_ENGINE'] = 'gpt-3.5-turbo-instruct'

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def generate_prompt(word_list: str):
    return f"""I would like you to act as an entity analyst. I provide you with an entity list, and I request that you select entities based on the following criteria:1. The entity should be visually observable by the human eye. 2. The entity represents a living organism, artifact, or natural landscape. 3.  

Entity list: tunicate, waterfowl, cisco, fluoridation, joker, Bellis, Anasazi, heliopsis, Micropogonias, mayoralty, shower curtain, outstroke, giardiasis, raita, Francophile, gulping, dugong, painted daisy, divulgence, pfennig, Vali, serviceman, orthopedist, Neoceratodus, mews, highness, fifth wheel, computerization, rack of lamb, Jamestown, Atriplex, rennet, secateurs, bearded seal, inguinal canal
Result: 
- waterfowl
- cisco
- joker 
- shower curtain
- dugong
- painted daisy
- serviceman
- orthopedist
- mews
- rack of lamb 
- secateurs
- bearded seal

Entity list: {word_list}.
Result:
- 
"""

def stringtolist(description):
    return [descriptor[2:] for descriptor in description.split('\n') if (descriptor != '') and (descriptor.startswith('- '))]

if __name__ == '__main__':
    
    save_root = 'GPT3/descriptors'
    dataset = 'wordnet'
    root_path = 'GPT3/descriptors'
    
    load_path = ''
    mode = 'initilize'
    
    if mode == 'initilize':
        # ==================== 1. WordNet ====================
        nltk.download('wordnet')  #  
        all_nouns = list(wordnet.all_synsets(pos=wordnet.NOUN))
        noun_names = [noun.lemmas()[0].name().replace('_', ' ') for noun in all_nouns]
        unique_noun_names = list(set(noun_names))
        word_num = len(unique_noun_names)
    else:
        with open(os.path.join(root_path, load_path), 'rb') as f:
            unique_noun_names = json.load(f)
    
    
    
    gpt_process_per_word = 100
    iter_step = 5
    
    for iter in range(2, iter_step+1):
        results = [] 
        word_num = len(unique_noun_names)
        for i in tqdm(range(word_num // gpt_process_per_word)):
            words = unique_noun_names[i*gpt_process_per_word:(i+1)*gpt_process_per_word]
            words = ', '.join(words)
            
            prompt = generate_prompt(words)
            responses = completion_with_backoff(model=os.environ['GPT_ENGINE'],
                                                    prompt=prompt,
                                                    temperature=0.,
                                                    max_tokens=300,
                                                    )
            response_texts = responses['choices'][0]['text']
            descriptors_list = stringtolist(response_texts)
            results.extend(descriptors_list)
        with open(os.path.join(save_root, 'descriptors_' + dataset + '_' + str(iter) + '.json'), 'w') as fp:
            json.dump(results, fp)
            
        unique_noun_names = results