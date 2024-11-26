import os
import json

hparams = {}
# hyperparameters
hparams['category_name_inclusion'] = 'prepend' #'append' 'prepend'
hparams['apply_descriptor_modification'] = True
hparams['verbose'] = False
hparams['before_text'] = ""
hparams['label_before_text'] = ""
hparams['between_text'] = ', '
# hparams['between_text'] = ' '
# hparams['between_text'] = ''
hparams['after_text'] = ''
hparams['unmodify'] = True
# hparams['after_text'] = '.'
# hparams['after_text'] = ' which is a type of bird.'
hparams['label_after_text'] = ''
# hparams['label_after_text'] = ' which is a type of bird.'

# unmodify_dict = {}

hparams['descriptor_fname'] = None

def load_json(filename):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)

def wordify(string):
    word = string.replace('_', ' ')
    return word

def make_descriptor_sentence(descriptor):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"

def modify_descriptor(descriptor, apply_changes):
    if apply_changes:
        return make_descriptor_sentence(descriptor)
    return descriptor

def load_gpt_descriptions(dataset, mode=None):
    
    gpt_descriptions_all = load_json(os.path.join('GPT3/descriptors', 'descriptors_{}.json'.format(dataset)))
    
    unmodify_dict = {}
    if not mode:
        return gpt_descriptions_all, None
    
   
    for i, (k, v) in enumerate(gpt_descriptions_all.items()):
        if len(v) == 0:
            v = ['']
        
        word_to_add = wordify(k)
            
        if mode == 'append':
            build_descriptor_string = lambda item: f"{modify_descriptor(item, hparams['apply_descriptor_modification'])}{hparams['between_text']}{word_to_add}"
        elif mode == 'prepend':
            build_descriptor_string = lambda item: f"{hparams['before_text']}{word_to_add}{hparams['between_text']}{modify_descriptor(item, hparams['apply_descriptor_modification'])}{hparams['after_text']}"
        else:
            build_descriptor_string = lambda item: modify_descriptor(item, hparams['apply_descriptor_modification'])
            
        unmodify_dict[k] = {build_descriptor_string(item): item for item in v}
                
        gpt_descriptions_all[k] = [build_descriptor_string(item) for item in v]
            
            # print an example the first time
        if i == 0: #verbose and 
            print(f"\nExample description for class {k}: \"{gpt_descriptions_all[k][0]}\"\n")
    return gpt_descriptions_all, unmodify_dict



if __name__ == '__main__':
    gpt_descriptions_split, unmodify_dict = load_gpt_descriptions('miniImageNet', 'train')
    