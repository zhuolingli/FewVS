from torch.utils.data import DataLoader
from dataloader.dataset import SMKD_dataset
from PIL import Image
import torch.distributed as dist
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torch
from torch.utils.data.sampler import Sampler
import numpy as np


class CategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.catlocs = []
        for i in range(min(label), max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.catlocs.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            choose_classes = torch.randperm(len(self.catlocs))[:self.n_cls]
            for c in choose_classes:
                l = self.catlocs[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class DCategoriesSampler(Sampler):
    def __init__(self, label, n_batch, n_cls, n_per, ep_per_batch=8,
                 num_replicas=None, rank=None):
        super().__init__(self)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank

        self.n_batch = n_batch  # batchs for each epoch
        self.n_cls = n_cls  # ways
        self.n_per = n_per  # shots
        self.num_samples = self.n_cls * self.n_per
        self.ep_per_batch = ep_per_batch
        label = np.array(label)
        self.catlocs = []
        for i in range(min(label), max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.catlocs.append(ind)

        self.epoch = 0

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                choose_classes = torch.randperm(len(self.catlocs))[:self.n_cls]
                for c in choose_classes:
                    l = self.catlocs[c]
                    samples = torch.randperm(len(l))[:self.n_per]
                    episode.append(l[samples])
                batch.append(torch.stack(episode).t().reshape(-1))
            batch = torch.stack(batch).reshape(-1)
            # subsample
            offset = self.num_samples * self.rank
            batch = batch[offset: offset + self.num_samples]
            # print(">" * 50, self.rank, len(batch), self.num_samples)
            assert len(batch) == self.num_samples

            yield batch



def get_smkd_dataloder(opt):
    trainset = SMKD_dataset(opt.dataset, backbone_type=opt.backbone, split="train")
    valset = SMKD_dataset(opt.dataset, backbone_type=opt.backbone, split="val")
    testset = SMKD_dataset(opt.dataset, backbone_type=opt.backbone, split="test")
    
    train_sampler = DCategoriesSampler(label=trainset.labels,
                                    n_batch=500,
                                    n_cls=opt.n_train_ways,
                                    n_per=opt.n_train_shots + opt.n_train_queries) if opt.distributed \
                else CategoriesSampler(label=trainset.labels,
                                        n_batch=500,
                                        n_cls=opt.n_train_ways,
                                        n_per=opt.n_train_shots + opt.n_train_queries)
    # 验证集和测试集默认使用单卡训练。
    val_sampler = CategoriesSampler(label=valset.labels,
                                        n_batch=opt.episodes,
                                        n_cls=opt.n_ways,
                                        n_per=opt.n_shots + opt.n_queries)
    test_sampler = CategoriesSampler(label=testset.labels,
                                    n_batch=opt.episodes,
                                    n_cls=opt.n_ways,
                                    n_per=opt.n_shots + opt.n_queries)
    if opt.mode == 'global':
        trainloader = DataLoader(dataset=trainset,
                                batch_size=opt.batch_size, shuffle=True,
                                num_workers=opt.num_workers, pin_memory=True)
    else:            
        trainloader = DataLoader(dataset=trainset,
                                num_workers=opt.num_workers,
                                batch_sampler=train_sampler,
                                pin_memory=True)
    valloader = DataLoader(dataset=valset,
                        num_workers=opt.num_workers,
                        batch_sampler=val_sampler,
                        pin_memory=False)
    testloader = DataLoader(dataset=testset,
                        num_workers=opt.num_workers,
                        batch_sampler=test_sampler,
                        pin_memory=False)
    return trainloader, valloader, testloader
