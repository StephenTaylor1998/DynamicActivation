import os
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from core import models
from core.datasets.data_zoo import get_data_by_name
from core.engine.auto_encoder.base import train, validate
from core.engine.base import save_checkpoint, adjust_learning_rate
from core.utils.copy_weights import copy_weights
from core.utils.resume import resume_from_checkpoint
# from core.datasets.data.image_folder import classify_train_dataset
# from core.datasets.data.image_folder import classify_val_dataset
# from core.datasets.data.image_folder import classify_test_dataset


def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, num_classes=args.classes)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=args.classes)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr,
    #                             weight_decay=args.weight_decay)

    # optionally resume from a checkpoint

    cudnn.benchmark = True

    # Data loading code
    train_dataset, val_dataset, test_dataset = get_data_by_name(args.data_format, data_dir=args.data)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ #

    # val data loader is here #
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    # ^^^^^^^^^^^^^^^^^^^^^^^ #
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.test:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        validate(train_loader, model, criterion, args)
        print('TEST IN TRAIN SET')
        validate(val_loader, model, criterion, args)
        print('TEST IN VAL SET')
        validate(test_loader, model, criterion, args)
        print('TEST IN TEST SET')
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        validate(val_loader, model, criterion, args)
