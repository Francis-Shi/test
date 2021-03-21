import os
import shutil
import time
import argparse

import numpy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist
import datetime

from models import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--sample_interval', default=20, type=int,
                    metavar='N', help='sample interval')

def coordinate(rank, world_size):
    args = parser.parse_args()
    current_lr = args.lr  # learning_rate
    sample_interval = args.sample_interval

    output_path ='ASGD_resnet20' + '_' + str(world_size) + '_lr_' + str(current_lr)
    os.mkdir(output_path)

    adjust = [80, 120]
    model = resnet20()
    model = model.cuda()
    model_flat = flatten_all(model)
    w_flat = flatten(model)
    g_flat = torch.zeros_like(w_flat)

    dim = g_flat.size(0)
    half = (dim - 1) // 2 + 1
    z_flat = torch.zeros(2 * half).cuda()
    z_flat_c = z_flat[0:half].char()
    delta = torch.FloatTensor(1).cuda()

    dist.broadcast(model_flat, world_size)

    cudnn.benchmark = True

    # Data loading code
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=25 * world_size, pin_memory=True, shuffle=False,
                                               num_workers=2)

    time_stamp = 0
    t1 = time.time()
    time_spend = []
    for epoch in range(args.epochs):
        print('server\'s epoch: '+str(epoch))

        # adjust learning rate

        if epoch in adjust:
            current_lr = current_lr * 0.1

        for i in range(len(train_loader)*world_size):
            print('server\'s iteration: '+str(i))

            dist.recv(delta, tag=111)
            src = dist.recv(z_flat_c, tag=222)

            z_flat[half:2 * half].copy_(z_flat_c.fmod(16))
            z_judge1 = (z_flat[half:2 * half] > 7).float()
            z_judge2 = (z_flat[half:2 * half] < -8).float()
            z_flat[half:2 * half].add_(-z_judge1 * 16).add_(z_judge2 * 16)
            z_flat[0:half].copy_(z_flat_c // 16)
            z_flat[0:half].add_(z_judge1).add_(-z_judge2)
            z_flat.mul_(delta)

            g_flat = z_flat[0:dim]

            w_flat.add_(-current_lr, g_flat)
            dist.send(w_flat, src, tag=333)

            time_stamp += 1
            print('time_stamp: ' + str(time_stamp))
            if time_stamp % sample_interval == 1:
                t2 = time.time()
                time_spend.append(t2-t1)
            time_stamp_t = torch.FloatTensor([time_stamp])
            dist.send(time_stamp_t, src, tag=444)

    print('training finished.')

    # test saved models
    output_file = open(output_path + '.txt', "w")

    dist.barrier()

    ite_len = len(time_spend)
    ite_len_t = torch.IntTensor([ite_len])
    dist.broadcast(ite_len_t, world_size)
    loss = torch.zeros(1)
    prec1 = torch.zeros(1)
    for ite in range(ite_len):
        loss.zero_()
        prec1.zero_()
        dist.reduce(loss, world_size)
        dist.reduce(prec1, world_size)
        loss.div_(world_size)
        prec1.div_(world_size)
        os.remove(output_path + '/' + str(ite * sample_interval + 1) + '.pth')
        output_file.write('%d %3f %3f %3f\n' % (ite*sample_interval+1, time_spend[ite], loss.numpy(), prec1.numpy()))
        output_file.flush()
    # close output file, stop
    output_file.close()
    os.removedirs(output_path)


def run(rank, world_size):
    args = parser.parse_args()
    current_lr = args.lr
    sample_interval = args.sample_interval

    print('Start node: %d  Total: %3d' % (rank, world_size))

    model = resnet20()
    model = model.cuda()
    model_flat = flatten_all(model)
    dist.broadcast(model_flat, world_size)
    # print('worker: '+str(rank)+'broadcast model_flat')

    unflatten_all(model, model_flat)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # Data loading code
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    val_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=25, pin_memory=True, shuffle=False,
                                               num_workers=2, sampler=train_sampler)

    output_path ='ASGD_resnet20' + '_' + str(world_size) + '_lr_' + str(current_lr)

    cur_time_stamp = [0]
    for epoch in range(args.epochs):
        print('worker\'s epoch: ' + str(epoch))
        # train for one epoch
        train_sampler.set_epoch(0)
        train(train_loader, model, criterion, epoch, rank, world_size, cur_time_stamp, output_path, sample_interval)


    # testing saved models
    valset = datasets.CIFAR10(root='./data', train=False, download=False, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=100, pin_memory=True, shuffle=False, num_workers=2)

    dist.barrier()

    ite_len = torch.IntTensor([0])
    dist.broadcast(ite_len, world_size)
    for ite in range(int(ite_len[0])):
        checkpoint = torch.load(output_path+'/'+str(ite*sample_interval+1)+'.pth')
        model.load_state_dict(checkpoint['model'])

        loss, _ = validate(train_loader, model, criterion)
        loss_t = torch.tensor([loss])
        _, prec1 = validate(val_loader, model, criterion)

        dist.reduce(loss_t, world_size)
        dist.reduce(prec1.cpu(), world_size)
        # dist.send(loss_t, world_size, tag=250)
        # dist.send(prec1.cpu(), world_size, tag=251)


def train(train_loader, model, criterion, epoch, rank, world_size, cur_time_stamp, output_path, sample_interval):
    wd = 0.0001
    # switch to train mode
    model.train()
    # cost = 0
    w_flat = flatten(model)
    g_flat = torch.zeros_like(w_flat)

    dim = g_flat.size(0)
    half = (dim - 1) // 2 + 1
    z_flat = torch.zeros(2 * half).cuda()
    p_flat = torch.zeros_like(z_flat)
    z_flat_c = z_flat[0:half].char()
    for i, (input, target) in enumerate(train_loader):

        input_var = torch.autograd.Variable(input.cuda())
        target = target.cuda()
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        model.zero_grad()
        loss.backward()
        flatten_g(model, g_flat)
        g_flat.add_(wd, w_flat)

        # encode
        z_flat[0:dim].copy_(g_flat)
        p_flat.uniform_()
        delta = z_flat.abs().max().div(7.0)
        delta.add_(1e-9)
        z_flat.div_(delta.item())
        z_flat.add_(p_flat)
        z_flat.floor_()
        z_flat_c.copy_(z_flat[0:half].mul(16))
        z_flat_c.add_(z_flat[half:2 * half].char())

        time_stamp_t = torch.zeros(1)

        # communicate
        dist.send(delta, world_size, tag=111)

        dist.send(z_flat_c, world_size, tag=222)

        dist.recv(w_flat, world_size, tag=333)

        dist.recv(time_stamp_t, world_size, tag=444)

        unflatten(model, w_flat)
        time_stamp = time_stamp_t[0].numpy()
        time_delay = time_stamp - cur_time_stamp[0]
        cur_time_stamp[0] = time_stamp
        # save model for testing
        if cur_time_stamp[0] % sample_interval == 1:
            state = {
                'model': model.state_dict(),
            }
            torch.save(state, output_path + '/' + str(int(cur_time_stamp[0])) + '.pth')
            print('save model at time_stamp:' + str(int(cur_time_stamp[0])))

def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input.cuda())
        target = target.cuda()
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

    return losses.avg, top1.avg


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def flatten_all(model):
    vec = []
    for param in model.parameters():
        vec.append(param.data.view(-1))
    for b in model.buffers():
        vec.append(b.data.float().view(-1))
    return torch.cat(vec)


def unflatten_all(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        param.data = vec[pointer:pointer + num_param].view(param.size())
        pointer += num_param
    for b in model.buffers():
        num_param = torch.prod(torch.LongTensor(list(b.size())))
        b.data = vec[pointer:pointer + num_param].view(b.size())
        pointer += num_param


def flatten(model):
    vec = []
    for param in model.parameters():
        vec.append(param.data.view(-1))
    return torch.cat(vec)


def unflatten(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        param.data = vec[pointer:pointer + num_param].view(param.size())
        pointer += num_param


def flatten_g(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        vec[pointer:pointer + num_param] = param.grad.data.view(-1)
        pointer += num_param


def unflatten_g(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        param.grad.data = vec[pointer:pointer + num_param].view(param.size())
        pointer += num_param


if __name__ == '__main__':
    dist.init_process_group('mpi')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # run(rank, world_size)

    if rank == world_size - 1:
        coordinate(rank, world_size - 1)
    else:
        run(rank, world_size - 1)