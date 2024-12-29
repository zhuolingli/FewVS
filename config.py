import argparse
server_dict = {
    'train_proj':{
        'miniImageNet': {
            'Res12': { 
                '1-shot': 'weights/miniImageNet/Res12/1shot.pth', 
                '5-shot': 'weights/miniImageNet/Res12/5shot.pth'},
            'SMKD': { 
                '1-shot': 'weights/miniImageNet/SMKD/1shot.pth',
                '5-shot': 'weights/miniImageNet/SMKD/5shot.pth'}},
        'tieredImageNet': {
            'Res12': { 
                '1-shot': 'weights/tieredImageNet/Res12/1shot.pth', 
                '5-shot': 'weights/tieredImageNet/Res12/5shot.pth'},
            'SMKD': { 
                '1-shot': 'weights/SMKD/Res12/1shot.pth', 
                '5-shot': 'weights/SMKD/Res12/5shot.pth'}},
        'FC100': {
            'Res12': {
                '1-shot': 'weights/FC100/Res12/1shot.pth', 
                '5-shot': 'weights/FC100/Res12/5shot.pth'},
            'SMKD': {
                '1-shot': 'weights/FC100/SMKD/1shot.pth', 
                '5-shot': 'weights/FC100/SMKD/5shot.pth'}},
        'CIFAR_FS': {
            'Res12': {
                '1-shot': 'weights/CIFAR_FS/Res12/1shot.pth', 
                '5-shot': 'weights/CIFAR_FS/Res12/5shot.pth'},
            'SMKD': {
                '1-shot': 'weights/CIFAR_FS/SMKD/1shot.pth', 
                '5-shot': 'weights/CIFAR_FS/SMKD/5shot.pth'},
        },
}
}  

def get_parser():
    parser = argparse.ArgumentParser(
        description='PyTorch Few-Shot Classification')
    # ================ arc ====================
    parser.add_argument("--snapshot_path", type=str, default="exp",)
    parser.add_argument("--mode", type=str,
                        choices=['train_proj', 'FewVS'], default="FewVS")
    parser.add_argument('--backbone', type=str, 
                        choices=["Res12", 'SMKD'], default="Res12")    
    parser.add_argument('--clip_backbone', type=str, 
                        choices=[ "RN50", "ViT-B/16", "RN50x4", "RN101"], default="RN50x4")
    parser.add_argument("--load_json_for_train_proj", type=str, 
                        default='imagenet_wo_mini')
                        # default='FC100')
                        # default='CIFAR_FS')
    parser.add_argument('--alpha', type=float, default=2)  
    parser.add_argument('--sem', type=float, default=1)  
    
    # ============= swav params ===============
    parser.add_argument("--epsilon", type=float, 
                        default=0.05, help="regularization parameter for Sinkhorn-Knopp algorithm")
    parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
    parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
    parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
    parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
    

    # =============== training ================
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--stop_interval', type=int, default=80)
    parser.add_argument('--eval_freq', type=int, default=5,
                        help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int,
                        default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int,
                        default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int,
                        default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--manual_seed', type=int,
                        default=0)
    parser.add_argument('--seed_deterministic', type=bool,
                        default=False)
    parser.add_argument('--fix_random_seed_val', type=bool,
                        default=True)
    # optimization
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument("--decay_step", type=int, default=40)
    parser.add_argument('--optim', type=str,
                        choices=["adam", "SGD", "adamw"], default="adamw")
    parser.add_argument('--w', type=list,
                        default=[1,0.,0.,0,]) # [swav, few-shot, zero-shot, mse]
    # dataset
    parser.add_argument('--dataset', type=str, 
                        default='tieredImageNet', choices=['miniImageNet', 'tieredImageNet', 'FC100','CIFAR_FS'])
    # meta setting
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=5, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_train_ways', type=int, default=15, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_train_shots', type=int, default=5, metavar='N',
                        help='Number of shots during train')
    parser.add_argument('--n_train_queries', type=int, default=7, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--episodes', default=2000, help="test episode num")
    # spatial for single view training
    parser.add_argument("--spatial", action="store_true")
    # continue train
    parser.add_argument("--resume", default=None)
    # test
    parser.add_argument("--is_test", action="store_true")
    
    # =============== online_optimizer ===================
    # FSL adaptation component related parameters
    parser.add_argument('--block_mask_1shot', default=5, type=int, help="""Number of patches to mask around each 
                        respective patch during online-adaptation in 1shot scenarios: masking along main diagonal,
                        e.g. size=5 corresponds to masking along the main diagonal of 'width' 5.""")

    parser.add_argument('--similarity_temp_init', type=float, default=0.051031036307982884,
                        help="""Initial value of temperature used for scaling the logits of the path embedding 
                            similarity matrix. Logits will be divided by that temperature, i.e. temp<1 scales up. 
                            'similarity_temp' must be positive.""")
    # Adaptation component -- Optimisation related parameters
    parser.add_argument('--optimiser_online', default='SGD', type=str, choices=['SGD'],
                        help="""Optimiser to be used for adaptation of patch embedding importance vector.""")
    parser.add_argument('--lr_online', default=0.05, type=float, help="""Learning rate used for online optimisation.""")
    parser.add_argument('--optim_steps_online', default=5, type=int, help="""Number of update steps to take to
                                optimise the patch embedding importance vector.""")
    
    # DistributedDataParallel
    parser.add_argument('--local_rank', type=int, default=0,
                        help='number of cpu threads to use during batch generation')
    
    parser.add_argument("--world_size", default=-1, type=int, 
                        help="number of processes: it is set automatically and should not be passed as argument")
    
    args = parser.parse_args()
    if args.n_shots == 1:
        load_shot_model = '1-shot'
    else:
        load_shot_model = '5-shot'
    args.test_weight = server_dict['train_proj'][args.dataset][args.backbone][load_shot_model]
    if args.dataset == "miniImageNet" or args.dataset == "CIFAR_FS":
        args.n_cls = 64
    elif args.dataset == "tieredImageNet":
        args.n_cls = 351
    elif args.dataset == 'FC100':
        args.n_cls = 60
    elif args.dataset == "CUB":
        args.n_cls = 100
    else:
        raise ValueError("wrong dataset")
    if args.optim == 'SGD':
        args.initial_lr = 1e-1
    else:
        args.initial_lr = 1e-3
    return args

