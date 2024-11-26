# FewVS 

This is the PyTorch implementation of "FewVS: A Vision-Semantics Integration Framework for Few-Shot Image Classification". 

## Installation

Python 3.8, Pytorch 1.11, CUDA 11.3. The code is tested on Ubuntu 20.04.


We have prepared `requirements.txt` file which contains all the python dependencies.

```sh
conda activate fewvs
pip install -r requirements.txt 
```

## Datasets

We prepare 𝒎𝒊𝒏𝒊ImageNet and 𝒕𝒊𝒆𝒓𝒆𝒅ImageNet and resize the images following the guidelines from [SMKD](https://github.com/HL-hanlin/SMKD). 

- **𝒎𝒊𝒏𝒊ImageNet**


> The 𝑚𝑖𝑛𝑖ImageNet dataset was proposed by [Vinyals et al.](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) for few-shot learning evaluation. Its complexity is high due to the use of ImageNet images but requires fewer resources and infrastructure than running on the full [ImageNet dataset](https://arxiv.org/pdf/1409.0575.pdf). In total, there are 100 classes with 600 samples of color images per class. These 100 classes are divided into 64, 16, and 20 classes respectively for sampling tasks for meta-training, meta-validation, and meta-test. To generate this dataset from ImageNet, you may use the repository [𝑚𝑖𝑛𝑖ImageNet tools](https://github.com/y2l/mini-imagenet-tools).

Note that in SMKD implementation, images are resized to 480 × 480 because the data augmentation used require the image resolution to be greater than 224 to avoid distortions. Therefore, when generating 𝒎𝒊𝒏𝒊ImageNet, you should set ```--image_resize 0``` to keep the original size or ```--image_resize 480``` as what we did.



- **𝒕𝒊𝒆𝒓𝒆𝒅ImageNet**

> The [𝑡𝑖𝑒𝑟𝑒𝑑ImageNet](https://arxiv.org/pdf/1803.00676.pdf) dataset is a larger subset of ILSVRC-12 with 608 classes (779,165 images) grouped into 34 higher-level nodes in the ImageNet human-curated hierarchy. To generate this dataset from ImageNet, you may use the repository 𝑡𝑖𝑒𝑟𝑒𝑑ImageNet dataset: [𝑡𝑖𝑒𝑟𝑒𝑑ImageNet tools](https://github.com/y2l/tiered-imagenet-tools). 

Similar to 𝒎𝒊𝒏𝒊ImageNet, you should set ```--image_resize 0``` to keep the original size or ```--image_resize 480``` as what we did when generating 𝒕𝒊𝒆𝒓𝒆𝒅ImageNet.


- **CIFAR-FS and FC100**

CIFAR-FS and FC100 can be download using  [CIFAR_FS](https://drive.google.com/file/d/1mdY1povHo9GPC6RA1upU-7ZKpNWCO6eO/view?usp=drive_link) and [FC100](https://drive.google.com/file/d/1SQiw2zr_viuZdaYi6JvyL3qri_TwnwDw/view?usp=sharing) or backup links ([CIFAR_FS](https://pan.baidu.com/s/15VfD_O5_t8Q3R21DNsxAOg?pwd=1111) and [FC100](https://pan.baidu.com/s/1DoRnS0q-7SNvwtPjF0-vMg?pwd=1111)).


**After getting the data, please put the four datasets into `data/`, and modify ``data_path`` in `dataloader/dataset.py` correspondingly.**

## Weights
- Please download the pre-trained models for evaluation from this [link](https://drive.google.com/file/d/1itn_U2zvnvRe841QBxK9NNvAOOYcT_xV/view?usp=sharing),  or using backup links ([miniImageNet](https://pan.baidu.com/s/1uDj5y7i_ticjlllYPVd6jw?pwd=1111), [tieredImageNet](https://pan.baidu.com/s/1x-J4dfqnguNqrBURk-kw6Q?pwd=1111), [FC100](https://pan.baidu.com/s/1tGag6VpME0u6xFnl1RGcjw?pwd=1111), [CIFAR_FS](https://pan.baidu.com/s/1gYjw0rOInhNG9jHwT_QJKw?pwd=1111), (code:1111)).

- unzip these weights to ``weights/``

## Training 
During training, only a linear layer is optimized.
For example, we can use the following code for training with the ResNet backbone on the CIFAR_FS dataset:

```sh
python train.py  --mode train_proj --backbone Res12 --dataset CIFAR_FS  
```






## Evaluation 
We use ```train.py``` to evaluate a trained model.

For example, we can use the following code to do 5-way 1-shot and 5-way 5-shot evaluation on FewVS-Res on the CIFAR_FS dataset:

- **1-shot**
```sh
python train.py --is_test --mode FewVS --backbone Res12 --dataset CIFAR_FS  --n_shots 1  --optim_steps_online 1 --alpha 2
```

- **5-shot**:
```sh
python train.py --is_test --mode FewVS --backbone Res12 --dataset CIFAR_FS  --n_shots 5  --optim_steps_online 5 --alpha 3
```

## Acknowlegements
Part of the code refers to [SMKD](https://github.com/HL-hanlin/SMKD), [BML](https://github.com/ZZQzzq/BML), and [this repo](https://github.com/sachit-menon/classify_by_description_release). Please check them for more details and features.

## Citation
When using code within this repository, please refer the following [paper](https://arxiv.org/abs/2108.12104) in your publications:
```
@inproceedings{li2024fewvs,
  title={FewVS: A Vision-Semantics Integration Framework for Few-Shot Image Classification},
  author={Li, Zhuoling and Wang, Yong and Li, Kaitong},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={1341--1350},
  year={2024}
}
```

