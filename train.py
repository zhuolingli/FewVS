import os
import datetime
import time
import cv2
import numpy as np
import argparse
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.distributed as dist
import torch.distributed.launch
import torch.optim as optim
from sklearn import metrics
import torch.nn.functional as F

# from tensorboardX import SummaryWriter

from util.util import AverageMeter, get_model_para_number, setup_seed, get_logger, get_save_path, check_makedirs

from dataloader.dataloader import get_smkd_dataloder
from model.proj import ProjBuilder
from model.FewVS import FewVSBuilder
from config import get_parser
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def prepare_label(args, split="test"):
    # prepare one-hot label
    if split in ["test", "val"]:
        label = torch.arange(
            args.n_ways, dtype=torch.int16).repeat(args.n_queries)
    else:
        label = torch.arange(args.n_train_ways, dtype=torch.int16).repeat(
            args.n_train_queries)
    label = label.type(torch.LongTensor)
    if torch.cuda.is_available():
        label = label.cuda()
    return label

def get_model(args):
    if args.mode == 'train_proj':
        model = ProjBuilder(args)
    elif args.mode == 'FewVS':
        model = FewVSBuilder(args)
    else:
        raise ValueError('Dont support {}'.format(args.mode))
    if args.optim == "adam":
        optimizer = torch.optim.Adam(params=model.parameters(),
                                    lr=args.initial_lr)
    elif args.optim == "SGD":
        optimizer = optim.SGD(model.parameters(),
                            lr=args.initial_lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.initial_lr)
    
    
    if hasattr(model, 'freeze_modules'):
        model.freeze_modules()

    get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)
    
    model.cuda()
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[
                                                        args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
    
    # Get model para.
    total_number, learnable_number = get_model_para_number(model)
    if main_process():
        print('Number of Parameters: %d ' % (total_number / 1))
        print('Number of Learnable Parameters: %d ' % (learnable_number / 1))

    time.sleep(2)
    return model, optimizer

def main_process():
    return not args.distributed or (args.distributed and (args.local_rank == 0))

def load_checkpoint(model, optimizer, weight_path, logger):    
    if os.path.isfile(weight_path):
        if main_process():
            logger.info(
                "=> loading test checkpoint '{}'".format(weight_path))
        checkpoint = torch.load(
            weight_path, map_location=torch.device('cpu'))
        new_param = checkpoint['state_dict']
        try:
            model.load_state_dict(new_param)
        except RuntimeError:                   
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            model.load_state_dict(new_param, strict=False)
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        if main_process():
            logger.info(
                "=> loaded checkpoint ({}) for testing".format(weight_path))
    else:
        assert False, "=> no checkpoint found at '{}'".format(weight_path)

def train_on_epoch(args, model, optimizer, criterion_Intra, epoch, train_loader):
    global best_epoch, keep_epoch, val_num
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    model.train()
    start = time.time()
    end = time.time()
    val_time = 0.
    max_iter = args.epochs * len(train_loader)

    for i, (samples, global_targets) in enumerate(train_loader):
        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1

        samples = samples.cuda(non_blocking=True)
        global_targets = global_targets.cuda(non_blocking=True)
        meta_targets = prepare_label(args, split=train_loader.dataset.split)
        
        logits = model(samples, split=train_loader.dataset.split)
        
        if args.mode == "local":
            targets = meta_targets
        else:
            assert args.mode == "global"
            targets = global_targets
        if args.spatial:
            loss = 0
            for i in range(logits.size(2)):
                loss += criterion_Intra(logits[..., i],
                                        targets) / logits.size(2)
        else:
            loss = criterion_Intra(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

     
        # ===================summery=====================
        batch_time.update(time.time() - end - val_time)
        loss_meter.update(loss.item(), 25)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(
            int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f}'.format(epoch, args.epochs, i + 1, len(train_loader),
                                                        batch_time=batch_time,
                                                        data_time=data_time,
                                                        remain_time=remain_time,
                                                        loss_meter=loss_meter))
    epoch_time = time.time() - start
    if main_process():
        logger.info(
            f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

@torch.no_grad()
def distributed_sinkhorn(out, args):
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    world_size = torch.distributed.get_world_size() if args.distributed else 1

    B = Q.shape[1] * world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q) 
    if args.distributed:
        dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if args.distributed:
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

def train_projection(args, model, optimizer, criterion_Intra, epoch, meta_trainloader):
    model.train()
    
    num_steps = len(meta_trainloader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    swav_meter = AverageMeter()
    vis_meter = AverageMeter()
    sem_meter = AverageMeter()
    l2_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    for idx, (samples, global_targets) in enumerate(meta_trainloader):
        if type(samples) is list:
            samples = [ i.cuda(non_blocking=True) for i in samples]
        else:
            samples = samples.cuda(non_blocking=True)
        global_targets = global_targets.cuda(non_blocking=True)
        meta_targets = prepare_label(args, split=meta_trainloader.dataset.split)
        selected_labelidx = global_targets[:args.n_train_ways].tolist()
        selected_classes = [
            meta_trainloader.dataset.label2class[idx] for idx in selected_labelidx]
        with torch.no_grad():
            if not args.distributed:
                w = model.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                model.prototypes.weight.copy_(w)
            else:
                w = model.module.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                model.module.prototypes.weight.copy_(w)
        
        # =================== forward =====================
        logits_vis, logits_sem, loss_l2, output = model(samples, selected_classes, False, split=meta_trainloader.dataset.split)
        
        loss_swav = 0    
        bs = int(output.shape[0] / 2)
        
        for i, crop_id in enumerate(args.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # get assignments
                q = distributed_sinkhorn(out, args)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                x = output[bs * v: bs * (v + 1)] / args.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss_swav += subloss / (np.sum(args.nmb_crops) - 1)
        loss_swav /= len(args.crops_for_assign)
        
        loss_vis = criterion_Intra(logits_vis, meta_targets)
        loss_sem = criterion_Intra(logits_sem, meta_targets)
        # =================== cal loss =====================
        loss = args.w[0]*loss_swav + args.w[1]*loss_vis + args.w[2]*loss_sem + args.w[3]*loss_l2
        # ===================backward=====================
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        # ===================summery=====================
        batch_time.update(time.time() - end)
        l2_meter.update(loss_l2.item(), 25)
        vis_meter.update(loss_vis.item(), 25)
        sem_meter.update(loss_sem.item(), 25)
        swav_meter.update(loss_swav.item(), 25)
        
        loss_meter.update(loss.item(), 25)
        end = time.time()

        if idx % args.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            if main_process():
                logger.info(
                    f'Train: [{epoch}/{args.epochs}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'loss_swav {swav_meter.val:.4f} ({swav_meter.avg:.4f})\t'
                    f'loss_vis {vis_meter.val:.4f} ({vis_meter.avg:.4f})\t'
                    f'loss_sem {sem_meter.val:.4f} ({sem_meter.avg:.4f})\t'
                    f'loss_l2 {l2_meter.val:.4f} ({l2_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    if main_process():
        logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

@torch.no_grad()
def validate(model, loader, args):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    
    if args.manual_seed is not None:
        args.manual_seed = args.manual_seed + args.local_rank
        setup_seed(args.manual_seed, args.seed_deterministic)
    
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    model.eval()
    end = time.time()
    criterion = torch.nn.CrossEntropyLoss()
    acc = []
    
    for idx, (images, global_labels) in enumerate(loader):
        if type(images) is list:
            images = [i.cuda(non_blocking=True) for i in images]
        else:
            images = images.cuda(non_blocking=True)
        # compute output
        meta_labels = prepare_label(args, split=loader.dataset.split)
        meta_labels = meta_labels.cuda(non_blocking=True)
        selected_labelidx = global_labels[:len(set(global_labels.tolist()))].tolist()
        selected_classes = [
            loader.dataset.label2class[idx] for idx in selected_labelidx]

        if args.mode in ['train_proj', 'FewVS']: 
            output = model(images, selected_classes, split=loader.dataset.split)
        else:
            output = model(images, split=loader.dataset.split)
        # measure accuracy and record loss
        loss = criterion(output, meta_labels)
        loss_meter.update(loss.item(), meta_labels.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % args.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if main_process():
                logger.info(
                    f'Test: [{idx}/{len(loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Mem {memory_used:.0f}MB')
        logits = torch.argmax(output, dim=1)
        logits = logits.detach().cpu().numpy()
        meta_labels = meta_labels.detach().cpu().numpy()
        acc.append(metrics.accuracy_score(meta_labels, logits))
    acc_list = [i * 100 for i in acc]
    ci95 = 1.96 * np.std(acc_list, axis=0) / np.sqrt(len(acc_list))


    if main_process():
        logger.info(
            f' * Acc on {args.n_ways} way-{args.n_shots} shot: {np.mean(acc_list):.3f}({ci95:.3f})')
    if args.is_test and main_process():
        os.makedirs("eval_results", exist_ok=True)
        with open(os.path.join("eval_results", args.dataset + "_{}_eval_results_{}way_{}shot.txt".format(args.mode, args.n_ways, args.n_shots)), "a") as f:
            f.write(str("{}: {}({:.3f})\n".format(args.manual_seed, np.mean(acc_list), ci95)))
        f.close()
    return np.mean(acc_list)

def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    args.distributed = False
    
    if main_process():
        print(args)
    
    if args.distributed:
        # Initialize Process Group
        dist.init_process_group(backend='nccl')
        print('args.local_rank: ', args.local_rank)
        torch.cuda.set_device(args.local_rank)
    # =========================initilize============================
    
    if args.manual_seed is not None:
        args.manual_seed = args.manual_seed + args.local_rank
        setup_seed(args.manual_seed, args.seed_deterministic)

    if main_process():
        logger.info("=> creating model ...")

    model, optimizer = get_model(args)
    lrstep1, lrstep2 = 50, 70  # LR decay step

    # if main_process():
    #     logger.info(model)

# ----------------------  DATASET  ----------------------
    if main_process():
        logger.info("=> loading datasets ...")
    train_loader, val_loader, test_loader = get_smkd_dataloder(args)
    
# ----------------------  TEST  ----------------------
    if args.is_test:  
        filename = args.test_weight
        load_checkpoint(model, None, filename, logger)
        
        validate(model, test_loader, args)
        return

# ----------------------  TRAINVAL  ----------------------
    global best_acc, best_epoch, keep_epoch, val_num
    best_acc = 0.
    best_epoch = 0
    keep_epoch = 0
    val_num = 0

    start_time = time.time()
    criterion_Intra = nn.CrossEntropyLoss()

    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        if keep_epoch == args.stop_interval:
            break

        keep_epoch += 1
        if args.distributed and args.mode == 'global':
            # train_loader.sampler.set_epoch(epoch)
            pass
        # ----------------------  TRAIN  ----------------------
        if args.mode == 'train_proj':
            train_projection(args, model, optimizer, criterion_Intra, epoch, train_loader)
     
        # save model for <resuming>
        if (epoch % args.save_freq == 0) and main_process():
            filename = args.snapshot_path + '/epoch_{}.pth'.format(epoch)
            logger.info('Saving checkpoint to: ' + filename)
            if osp.exists(filename):
                os.remove(filename)
            state_dict = model.state_dict()
            filtered_state_dict = {key: value for key, value in state_dict.items() if "enc_t" not in key}
            torch.save({'epoch': epoch, 'state_dict': state_dict,
                'optimizer': optimizer.state_dict()}, filename)
    

        # -----------------------  VAL  -----------------------
        if epoch % args.eval_freq == 0:
            acc = validate(model, val_loader, args)
            val_num += 1
            best_acc = max(best_acc, acc)

            if main_process():
                logger.info(
                    f"Accuracy of the network : {acc:.2f}%")
                logger.info(f'Best accuracy: {best_acc:.2f}%')

        # save model for <testing>
            if acc == best_acc:
                best_acc, best_epoch = acc, epoch
                keep_epoch = 0
                filename = args.snapshot_path + \
                    '/train_epoch_{}'.format(epoch) + \
                    '_{:.4f}'.format(best_acc) + '.pth'
                if main_process():
                    logger.info('Saving checkpoint to: ' + filename)
                    state_dict = model.state_dict()
                    filtered_state_dict = {key: value for key, value in state_dict.items() if "enc_t" not in key}
                    torch.save({'epoch': epoch, 'state_dict': filtered_state_dict,
                            'optimizer': optimizer.state_dict()}, filename)

        if args.dataset == "tieredImageNet":
            if epoch % args.decay_step == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
                    print('-------Decay Learning Rate to ',
                          param_group['lr'], '------')
        else:
            decay1 = 0.06 if args.optim == 'SGD' else 0.1
            decay2 = 0.2 if args.optim == 'SGD' else 0.1

            if epoch == lrstep1:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= decay1
                    print('-------Decay Learning Rate to ',
                          param_group['lr'], '------')
            if epoch == lrstep2:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= decay2
                    print('-------Decay Learning Rate to ',
                          param_group['lr'], '------')

    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    if main_process():
        print('\nEpoch: {}/{} \t Total running time: {}'.format(epoch,
              args.epochs, total_time))
        print('The number of models validated: {}'.format(val_num))
        print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  Final Best Result   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print(args.mode +
              '\t Best_step:{}'.format(best_epoch))
        print('>'*80)
        print('%s' % datetime.datetime.now())

if __name__ == '__main__':
    main()
    # os.system("/usr/bin/shutdown")