# This code is built from the PyTorch examples repository: https://github.com/pytorch/examples/.
# Copyright (c) 2017 Torch Contributors.
# The Pytorch examples are available under the BSD 3-Clause License.
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2019 Adobe. All rights reserved.
# Adobe’s modifications are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License (CC-NC-SA-4.0). To view a copy of the license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
#
# ==========================================================================================
#
# BSD-3 License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE

import argparse
import os
import random
import shutil
import time
import warnings
import sys
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import antialiased_cnns
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/datasets01/imagenet_full_size/061417',
                    help='path to dataset')
parser.add_argument('--dryrun', action='store_true',
                    help='run on a mini dataset so you don\'t have to build the datafolders over the full imagenet \
                    dataset')
parser.add_argument('-l', '--learned-frame', action='store_true',
                    help='If true, we are going to learn a frame by gradient descent on the loss')
parser.add_argument('--entropy-scale', default=0.0, type=float,
                    metavar='ES', help='scale of entropy penalty on the learnt frame')
parser.add_argument('--num_samples', type=int, default=4, help='number of group samples from the frame')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-ep', '--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_step', default=30, type=float,
                    help='number of epochs before stepping down learning rate')
parser.add_argument('--cos_lr', action='store_true',
                    help='use cosine learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--force_nonfinetuned', dest='force_nonfinetuned', action='store_true',
                    help='if pretrained, load the model that is pretrained from scratch (if available)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--evaluate-save', dest='evaluate_save', action='store_true',
                    help='save validation images off')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# Added functionality from PyTorch codebase
parser.add_argument('--no-data-aug', dest='no_data_aug', action='store_true',
                    help='no shift-based data augmentation')
parser.add_argument('--out-dir', dest='out_dir', default='./', type=str,
                    help='output directory')
parser.add_argument('-es', '--evaluate-shift', dest='evaluate_shift', action='store_true',
                    help='evaluate model on shift-invariance')
parser.add_argument('--epochs-shift', default=5, type=int, metavar='N',
                    help='number of total epochs to run for shift-invariance test')
parser.add_argument('-ed', '--evaluate-diagonal', dest='evaluate_diagonal', action='store_true',
                    help='evaluate model on diagonal')
parser.add_argument('-ba', '--batch-accum', default=1, type=int,
                    metavar='N',
                    help='number of mini-batches to accumulate gradient over before updating (default: 1)')
parser.add_argument('--embed', dest='embed', action='store_true',
                    help='embed statement before anything is evaluated (for debugging)')
parser.add_argument('--val-debug', dest='val_debug', action='store_true',
                    help='debug by training on val set')
parser.add_argument('--weights', default=None, type=str, metavar='PATH',
                    help='path to pretrained model weights')
parser.add_argument('--save_weights', default=None, type=str, metavar='PATH',
                    help='path to save model weights')
parser.add_argument('--finetune', action='store_true', help='finetune from baseline model')
parser.add_argument('-mti', '--max-train-iters', default=np.inf, type=int,
                    help='number of training iterations per epoch before cutting off (default: infinite)')

parser.add_argument('--wandb', action='store_true', help='use wandb logging')
best_acc1 = 0


def main():
    args = parser.parse_args()

    if(not os.path.exists(args.out_dir)):
        os.mkdir(args.out_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    print("=> creating model '{}'".format(args.arch))
    if(args.arch.split('_')[-1][:-1]=='lpf'): # antialiased model
        model = antialiased_cnns.__dict__[args.arch[:-5]](pretrained=args.pretrained, 
                                                          filter_size=int(args.arch[-1]), 
                                                          _force_nonfinetuned=args.force_nonfinetuned)
    else: # baseline model
        model = models.__dict__[args.arch](pretrained=args.pretrained)

    # instrumentation
    if(args.wandb):
        import wandb
        wandb.init(project='antialiased-cnns')
        wandb.config.update(args)
        wandb.watch(model)

    if args.finetune: # finetune from baseline "aliased" model
        print("=> copying over pretrained weights from [%s]"%args.arch[:-5])
        model_baseline = models.__dict__[args.arch[:-5]](pretrained=True)
        antialiased_cnns.copy_params_buffers(model_baseline, model)

    if args.weights is not None:
        print("=> using saved weights [%s]"%args.weights)
        weights = torch.load(args.weights)
        model.load_state_dict(weights['state_dict'])

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
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

    if args.learned_frame:
        frame = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=3,
                kernel_size=7,
                padding='same', padding_mode='circular',
                dilation=3,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=3, out_channels=1,
                kernel_size=7,
                padding='same', padding_mode='circular',
                dilation=3,
            ),
            nn.Flatten(),
            nn.LogSoftmax(dim=-1),
            nn.Unflatten(dim=-1,unflattened_size=[224, 224])
        )
        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                frame.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int(args.workers / ngpus_per_node)
                frame = torch.nn.parallel.DistributedDataParallel(frame, device_ids=[args.gpu])
            else:
                frame.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                frame = torch.nn.parallel.DistributedDataParallel(frame)
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            frame = frame.cuda(args.gpu)
        else:
            frame = torch.nn.DataParallel(frame).cuda()
        # TODO(eugenevinitsky) add an option for fine-tuning the model as well
        optimizer = torch.optim.SGD(frame.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        model.requires_grad = False
    else:
        frame = None
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
     # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            if('optimizer' in checkpoint.keys()): # if no optimizer, then only load weights
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                if args.gpu is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = best_acc1.to(args.gpu)
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('  No optimizer saved')
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    if args.dryrun:
        tempdir = '/checkpoint/eugenevinitsky/imagenet_subsample'
        traindir = os.path.join(tempdir, 'train')
        valdir = os.path.join(tempdir, 'val')
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    if(args.no_data_aug):
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    crop_size = 256 if(args.evaluate_shift or args.evaluate_diagonal or args.evaluate_save) else 224
    args.batch_size = 1 if (args.evaluate_diagonal or args.evaluate_save) else args.batch_size

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if(args.val_debug): # debug mode - train on val set for faster epochs
        train_loader = val_loader

    if(args.embed):
        from IPython import embed
        embed()

    if args.save_weights is not None: # "deparallelize" saved weights
        print("=> saving 'deparallelized' weights [%s]"%args.save_weights)
        # TO-DO: automatically save this during training
        if args.gpu is not None:
            torch.save({'state_dict': model.state_dict()}, args.save_weights, _use_new_zipfile_serialization=False)
        else:
            if(args.arch[:7]=='alexnet' or args.arch[:3]=='vgg'):
                model.features = model.features.module
                torch.save({'state_dict': model.state_dict()}, args.save_weights, _use_new_zipfile_serialization=False)
            else:
                torch.save({'state_dict': model.module.state_dict()}, args.save_weights, _use_new_zipfile_serialization=False)
        return

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if(args.evaluate_shift):
        validate_shift(val_loader, model, args)
        return

    if(args.evaluate_diagonal):
        validate_diagonal(val_loader, model, args)
        return

    if(args.evaluate_save):
        validate_save(val_loader, mean, std, args)
        return

    if(args.cos_lr):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        for epoch in range(args.start_epoch):
            scheduler.step()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if(not args.cos_lr):
            adjust_learning_rate(optimizer, epoch, args)
        else:
            scheduler.step()
            print('[%03d] %.5f'%(epoch, scheduler.get_lr()[0]))

        if(args.wandb):
            wandb.log({'learning_rate': optimizer.param_groups[0]['lr']},
                      commit=False)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, frame)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, epoch, out_dir=args.out_dir)


def train(train_loader, model, criterion, optimizer, epoch, args, frame=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    output_device = next(model.parameters()).device
    # switch to train mode
    model.train()

    end = time.time()
    accum_track = 0
    optimizer.zero_grad()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # TODO(eugenevinitsky) I think 
        # compute output
        if frame is None:
            output = model(input)
        else:
            input = input.to(output_device)
            frame_x = frame(input)
            cat = torch.distributions.categorical.Categorical(logits=frame_x.view(input.shape[0], -1))
            # now draw a few samples from the frame map and 
            frame_phis = torch.zeros(args.num_samples, input.shape[0], 1000, dtype=input.dtype).cuda(args.gpu, non_blocking=True)
            shift_imgs = torch.zeros_like(input, dtype=input.dtype).cuda(args.gpu, non_blocking=True)
            # TODO(eugenevinitsky) remove the double four loop
            for j in range(args.num_samples):
                sample = cat.sample()
                for k in range(input.shape[0]):
                    p = sample[k] % 224
                    q = sample[k] // 224
                    shift_imgs[k] = inv_shift(input[k], (p,q))
                frame_phis[j] = model(shift_imgs).detach() * torch.exp(cat.log_prob(sample) - cat.log_prob(sample).detach()).unsqueeze(1)
            output = frame_phis.mean(dim=0)
        loss = criterion(output, target)
        if frame is not None and args.entropy_scale > 0.0:
            loss = loss -args.entropy_scale * torch.sum(frame_x.exp()*frame_x+1e-6)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        accum_track+=1
        if(accum_track==args.batch_accum):
            optimizer.step()
            accum_track = 0
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

            if(args.wandb):
                import wandb
                global_step = i + (epoch * len(train_loader))
                wandb.log(
                    {
                        'train_loss': losses.val,
                        'train_avg_loss': losses.avg,
                        'train_acc@1': top1.val,
                        'train_avg_acc@1': top1.avg,
                        'train_acc@5': top5.val,
                        'train_avg_acc@5': top5.avg,
                        'epoch': 1.*global_step/len(train_loader), 
                    },
                    step=global_step)

        if(i > args.max_train_iters):
            break

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        if args.wandb:
            import wandb
            wandb.log(
                {
                    'val_avg_loss': losses.avg,
                    'val_avg_acc@1': top1.avg,
                    'val_avg_acc@5': top5.avg
                },
                commit=False)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def validate_shift(val_loader, model, args):
    batch_time = AverageMeter()
    consist = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for ep in range(args.epochs_shift):
            for i, (input, target) in enumerate(val_loader):
                if args.gpu is not None:
                    input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

                off0 = np.random.randint(32,size=2)
                off1 = np.random.randint(32,size=2)

                output0 = model(input[:,:,off0[0]:off0[0]+224,off0[1]:off0[1]+224])
                output1 = model(input[:,:,off1[0]:off1[0]+224,off1[1]:off1[1]+224])

                cur_agree = agreement(output0, output1).type(torch.FloatTensor).to(output0.device)

                # measure agreement and record
                consist.update(cur_agree.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Ep [{0}/{1}]:\t'
                          'Test: [{2}/{3}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Consist {consist.val:.4f} ({consist.avg:.4f})\t'.format(
                           ep, args.epochs_shift, i, len(val_loader), batch_time=batch_time, consist=consist))

        print(' * Consistency {consist.avg:.3f}'
              .format(consist=consist))

    return consist.avg

def validate_diagonal(val_loader, model, args):
    batch_time = AverageMeter()
    prob = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    D = 33
    diag_probs = np.zeros((len(val_loader.dataset),D))
    diag_probs2 = np.zeros((len(val_loader.dataset),D)) # save highest probability, not including ground truth
    diag_corrs = np.zeros((len(val_loader.dataset),D))
    diag_preds = np.zeros((len(val_loader.dataset),D))

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            inputs = []
            for off in range(D):
                inputs.append(input[:,:,off:off+224,off:off+224])
            inputs = torch.cat(inputs, dim=0)
            probs = torch.nn.Softmax(dim=1)(model(inputs))
            preds = probs.argmax(dim=1).cpu().data.numpy()
            corrs = preds == target.item()
            outputs = 100.*probs[:,target.item()]
            
            acc1, acc5 = accuracy(probs, target.repeat(D), topk=(1, 5))

            probs[:,target.item()] = 0
            probs2 = 100.*probs.max(dim=1)[0].cpu().data.numpy()

            diag_probs[i,:] = outputs.cpu().data.numpy()
            diag_probs2[i,:] = probs2
            diag_corrs[i,:] = corrs
            diag_preds[i,:] = preds

            # measure agreement and record
            prob.update(np.mean(diag_probs[i,:]), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Prob {prob.val:.4f} ({prob.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, prob=prob, top1=top1, top5=top5))

    print(' * Prob {prob.avg:.3f} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(prob=prob,top1=top1, top5=top5))

    np.save(os.path.join(args.out_dir,'diag_probs'),diag_probs)
    np.save(os.path.join(args.out_dir,'diag_probs2'),diag_probs2)
    np.save(os.path.join(args.out_dir,'diag_corrs'),diag_corrs)
    np.save(os.path.join(args.out_dir,'diag_preds'),diag_preds)

def validate_save(val_loader, mean, std, args):
    import matplotlib.pyplot as plt
    import os
    for i, (input, target) in enumerate(val_loader):
        img = (255*np.clip(input[0,...].data.cpu().numpy()*np.array(std)[:,None,None] + mean[:,None,None],0,1)).astype('uint8').transpose((1,2,0))
        plt.imsave(os.path.join(args.out_dir,'%05d.png'%i),img)

# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
def save_checkpoint(state, is_best, epoch, out_dir='./'):
    torch.save(state, os.path.join(out_dir,'checkpoint.pth.tar'))
    if(epoch % 10 == 0):
        torch.save(state, os.path.join(out_dir,'checkpoint_%03d.pth.tar'%epoch))
    if is_best:
        shutil.copyfile(os.path.join(out_dir,'checkpoint.pth.tar'), os.path.join(out_dir,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def agreement(output0, output1):
    pred0 = output0.argmax(dim=1, keepdim=False)
    pred1 = output1.argmax(dim=1, keepdim=False)
    agree = pred0.eq(pred1)
    agree = 100.*torch.mean(agree.type(torch.FloatTensor).to(output0.device))
    return agree

def shift(x, pq):
    if isinstance(pq, tuple) or isinstance(pq, list):
        p,q = pq
    else:
        p = q = pq
    assert p > 0 and q > 0
    y = torch.zeros(3, 224, 224, dtype=x.dtype)
    y[:, p:,q:] = x[:, :-p,:-q]
    y[:, :p,:q] = x[:, -p:,-q:]
    y[:, p:,:q] = x[:, :-p,-q:]
    y[:, :p,q:] = x[:, -p:,:-q]
    return y

def inv_shift(x, pq):
    if isinstance(pq, tuple) or isinstance(pq, list):
        p,q = pq
    else:
        p = q = pq
    return shift(x, (224-p, 224-q))

def save_image(x):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(x.cpu().numpy())
    plt.savefig('test.png')


if __name__ == '__main__':
    main()
