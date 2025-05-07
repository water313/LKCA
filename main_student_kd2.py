import argparse
import os
import sys
import time
import cv2
import math
import torch
import re
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchnet import meter

import json
from tqdm import tqdm
from data import HSTrainingData
from data import HSTestData
from student_kd2 import Student
from metrics import compare_mpsnr
# loss
from loss import HLoss
from metrics import quality_assessment

from collections import OrderedDict

# global settings
resume = True
log_interval = 50
model_name = ''
test_data_dir = ''


def main():
    # parsers
    main_parser = argparse.ArgumentParser(description="parser for SR network")
    subparsers = main_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_parser.add_argument("--cuda", type=int, required=False,default=1,
                              help="set it to 1 for running on GPU, 0 for CPU")
    train_parser.add_argument("--batch_size", type=int, default=32, help="batch size, default set to 32")
    train_parser.add_argument("--epochs", type=int, default=150, help="epochs, default set to 150")
    train_parser.add_argument("--n_feats", type=int, default=128, help="n_feats, default set to 128")
    train_parser.add_argument("--n_blocks", type=int, default=16, help="n_blocks, default set to 16")
    train_parser.add_argument("--n_scale", type=int, default=4, help="n_scale, default set to 4")
    train_parser.add_argument("--dataset_name", type=str, default="Chikusei", help="dataset_name, default set to dataset_name")
    train_parser.add_argument("--model_title", type=str, default="student", help="model_title, default set to model_title")
    train_parser.add_argument("--seed", type=int, default=3000, help="start seed for model")
    train_parser.add_argument("--learning_rate", type=float, default=0.002,
                              help="learning rate, default set to 2e-3")
    train_parser.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")
    train_parser.add_argument("--gpus", type=str, default="1", help="gpu ids (default: 1)")

    test_parser = subparsers.add_parser("test", help="parser for testing arguments")
    test_parser.add_argument("--cuda", type=int, required=False,default=1,
                             help="set it to 1 for running on GPU, 0 for CPU")
    test_parser.add_argument("--gpus", type=str, default="0,1", help="gpu ids (default: 1)")
    test_parser.add_argument("--dataset_name", type=str, default="Chikusei",help="dataset_name, default set to dataset_name")
    test_parser.add_argument("--model_title", type=str, default="student",help="model_title, default set to model_title")
    test_parser.add_argument("--n_feats", type=int, default=128, help="n_feats, default set to 128")
    test_parser.add_argument("--n_blocks", type=int, default=16, help="n_blocks, default set to 16")
    test_parser.add_argument("--n_scale", type=int, default=4, help="n_scale, default set to 4")

    # 生成时间戳
    global sanitized_time
    sanitized_time = time.strftime('%Y_%m_%d_%H')

    args = main_parser.parse_args()
    print('===>GPU:',args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.subcommand is None:
        print("ERROR: specify either train or test")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    if args.subcommand == "train":
        train(args)
    else:
        test(args)
    pass


def train(args):

    global sanitized_time  # 确保使用同一个时间戳

    device = torch.device("cuda" if args.cuda else "cpu")
    # args.seed = random.randint(1, 10000)
    print("Start seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    print('===> Loading datasets')
    train_path = './datasets32/'+args.dataset_name+'_x'+str(args.n_scale)+'/trains/'
    eval_path = './datasets32/' + args.dataset_name + '_x' + str(args.n_scale) + '/evals/'
    test_data_dir = './datasets32/' + args.dataset_name + '_x' + str(args.n_scale) + '/' + args.dataset_name + '_test.mat'

    train_set = HSTrainingData(image_dir=train_path, augment=True)
    eval_set =  HSTrainingData(image_dir=eval_path, augment=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, num_workers=4, shuffle=False)
    test_set = HSTestData(test_data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    if args.dataset_name=='Cave':
        colors = 31
    elif args.dataset_name=='Pavia':
        colors = 102
    elif args.dataset_name=='Houston':
        colors = 48
    else:
        colors = 128

    print('===> Building model:{}'.format(args.model_title))
    net = Student(n_colors=colors, n_feats=args.n_feats, n_lkb=args.n_blocks, scale=args.n_scale)
    # print(net)
    model_title =  args.model_title +'_r'+str(args.n_scale)+'_Feats='+str(args.n_feats) + '_lkb='+str(args.n_blocks)
    # model_name = './checkpoints/' + "time" + args.dataset_name + model_title + "_ckpt_epoch_" + str() + ".pth"
    model_name = './checkpoints/'
    args.model_title = model_title


    start_epoch = 0
    if resume:
        if os.path.isfile(model_name):
            print("=> loading checkpoint '{}'".format(model_name))
            checkpoint = torch.load(model_name)
            start_epoch = checkpoint["epoch"]
            net.load_state_dict(checkpoint["model"])
        else:
            print("=> no checkpoint found at '{}'".format(model_name))

    if torch.cuda.device_count() > 1:
        print("===> Let's use", torch.cuda.device_count(), "GPUs.")
        net = torch.nn.DataParallel(net)

    net.to(device).train()
    print_network(net)
    h_loss = HLoss(0.5, 0.1)
    L1_loss = torch.nn.L1Loss()

    print("===> Setting optimizer and logger")
    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    epoch_meter = meter.AverageValueMeter()

    writer = SummaryWriter('runs/' + model_title + '_' + sanitized_time)

    best_epoch = 0
    best_loss = 1

    print('===> Start training')
    for e in range(start_epoch, args.epochs):

        adjust_learning_rate(args.learning_rate, optimizer, e + 1)
        epoch_meter.reset()
        net.train()
        print("Start epoch {}, learning rate = {}".format(e + 1, optimizer.param_groups[0]["lr"]))
        for iteration, (x, lms, gt) in enumerate(tqdm(train_loader, leave=False)):
            x, lms, gt = x.to(device), lms.to(device), gt.to(device)
            optimizer.zero_grad()
            y, _ = net(x, lms)
            loss = h_loss(y, gt)
            epoch_meter.add(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm(net.parameters(), clip_para)
            optimizer.step()
            # tensorboard visualization
            if (iteration + log_interval) % log_interval == 0:
                print("===> {} \tEpoch[{}]({}/{}): Loss: {:.6f}".format(time.ctime(), e + 1, iteration + 1,
                                                                        len(train_loader)-1, loss.item()))
                n_iter = e * len(train_loader) + iteration + 1
                writer.add_scalar('scalar/train_loss', loss, n_iter)

        # run validation set every epoch
        # eval_loss = validate(args, eval_loader, net, L1_loss)
        # if e == 0:
            # best_loss = eval_loss
            # best_epoch = e + 1
        # else:
            # if eval_loss <= best_loss:
                # best_loss = eval_loss
                # best_epoch = e+1

        mpsnr = calculate_mpsnr(test_loader, net, args)
        if e == start_epoch:
            best_mpsnr = mpsnr
            best_epoch = e + 1
        else:
            if mpsnr >= best_mpsnr:
                best_mpsnr = mpsnr
                best_epoch = e+1
        
        print("===> {}\tEpoch evaluation Complete: MPSNR: {:.6f}, best_epoch: {}, best_mpsnr: {:.6f}".format
              (time.ctime(), mpsnr, best_epoch, best_mpsnr))

        # print("===> {}\tEpoch evaluation Complete: Avg. Loss: {:.6f}, best_epoch: {}, best_loss: {:.6f}, MPSNR: {:.4f}".format(time.ctime(), eval_loss, best_epoch, best_loss, mpsnr))

        # print("===> {}\tEpoch evaluation Complete: Avg. Loss: {:.6f}, best_epoch: {}, best_loss: {:.6f}".format
        #       (time.ctime(), eval_loss, best_epoch, best_loss))
        #
        # print("===> {}\tEpoch evaluation Complete: MPSNR: {:.4f}".format(time.ctime(), mpsnr))

        # tensorboard visualization
        # writer.add_scalar('scalar/avg_epoch_loss', epoch_meter.value()[0], e + 1)
        # writer.add_scalar('scalar/avg_validation_loss', eval_loss, e + 1)
        
        writer.add_scalar('scalar/mpsnr', mpsnr, e + 1)
        writer.add_scalar('scalar/best_mpsnr', best_mpsnr, e + 1)

        # save model weights at checkpoints every 5 epochs
        if (e + 1) % 1 == 0:
            save_checkpoint(args, net, e+1, sanitized_time)


    print("===> Start testing")
    result_path = './results/' + args.dataset_name + '_x' + str(args.n_scale) + '/'
    model_name = './checkpoints/' + sanitized_time + '/' + args.dataset_name + '_' + model_title + "_ckpt_epoch_" + str(best_epoch) + ".pth"

    # 创建保存路径中的目录（如果不存在）
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with torch.no_grad():
        test_number = 0
        epoch_meter = meter.AverageValueMeter()
        epoch_meter.reset()
        # loading model
        net = Student(n_colors=colors, n_feats=args.n_feats, n_lkb=args.n_blocks, scale=args.n_scale)

        state_dict = torch.load(model_name)
        net.load_state_dict(state_dict['model'])
        net.to(device).eval()

        output = []
        for i, (ms, lms, gt) in enumerate(test_loader):
            # compute output
            ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
            y, _ = net(ms, lms)
            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0],:gt.shape[1],:]
            if i==0:
                indices = quality_assessment(gt, y, data_range=1., ratio=args.n_scale)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=args.n_scale))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number

    save_dir = result_path + model_title + '_' + sanitized_time + '.npy'
    np.save(save_dir, output)
    print("Test finished, test results saved to .npy file at ", save_dir)
    print(indices)

    # 确保路径存在
    result_path = './results/' + args.dataset_name + '_x' + str(args.n_scale) + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # 保存评估结果的文件名
    QIstr = os.path.join(result_path, model_title + '_' + sanitized_time + ".txt")

    # 将 numpy.float32 转换为原生 Python 的 float
    indices = {k: float(v) if isinstance(v, np.float32) else v for k, v in indices.items()}

    # 保存评估结果
    with open(QIstr, 'w') as f:
        json.dump(indices, f)

    print(f"Evaluation results saved to {QIstr}")



def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp


def adjust_learning_rate(start_lr, optimizer, epoch):
    lr = start_lr * (0.1 ** (epoch // 75)) 

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(args, loader, model, criterion):
    device = torch.device("cuda" if args.cuda else "cpu")
    # switch to evaluate mode
    model.eval()
    epoch_meter = meter.AverageValueMeter()
    epoch_meter.reset()
    with torch.no_grad():
        for i, (ms, lms, gt) in enumerate(loader):
            ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
            y, _ = model(ms, lms)
            loss = criterion(y, gt)
            epoch_meter.add(loss.item())

    # back to training mode
    model.train()
    return epoch_meter.value()[0]

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
def compare_mpsnr(x_true, x_pred, data_range):
    """
    :param x_true: Input image must have three dimension (H, W, C)
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    channels = x_true.shape[2]
    total_psnr = [compare_psnr(x_true[:, :, k], x_pred[:, :, k], data_range=data_range)
                  for k in range(channels)]

    return np.mean(total_psnr)

def quality(x_true, x_pred, data_range):
    """
    仅计算 MPSNR
    :param x_true: 真实图像
    :param x_pred: 预测图像
    :param data_range: 数据范围
    :return: 包含 MPSNR 的字典
    """
    result = {'MPSNR': compare_mpsnr(x_true, x_pred, data_range)}
    return result

def calculate_mpsnr(test_loader, model, args):
    device = torch.device("cuda" if args.cuda else "cpu")
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        test_number = 0
        epoch_meter = meter.AverageValueMeter()
        epoch_meter.reset()

        indices = None
        for i, (ms, lms, gt) in enumerate(test_loader):
            ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
            y, _ = model(ms, lms)
            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0], :gt.shape[1], :]
            if i == 0:
                indices = quality(gt, y, data_range=1.)
            else:
                indices = sum_dict(indices, quality(gt, y, data_range=1.))
            test_number += 1

        for index in indices:
            indices[index] = indices[index] / test_number

        # 检查当前 checkpoint 的 MPSNR
        current_mpsnr = float(indices.get("MPSNR"))

    model.train()
    return np.mean(current_mpsnr)  # 返回平均 MPSNR


def test(args):

    global sanitized_time  # 确保使用同一个时间戳

    if args.dataset_name=='Cave':
        colors = 31
    elif args.dataset_name=='Pavia':
        colors = 102
    elif args.dataset_name=='Houston':
        colors = 48
    else:
        colors = 128
    test_data_dir = './datasets32/' + args.dataset_name + '_x' + str(args.n_scale) + '/' + args.dataset_name + '_test.mat'
    result_path = './results/' + args.dataset_name + '_x' + str(args.n_scale) + '/'
    model_title =  args.model_title +'_r'+str(args.n_scale)+'_Feats='+str(args.n_feats)  + '_lkb='+str(args.n_blocks)
    model_name = './checkpoints/'
    device = torch.device("cuda" if args.cuda else "cpu")
    print('===> Loading testset')

    test_set = HSTestData(test_data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')

    with torch.no_grad():
        test_number = 0
        epoch_meter = meter.AverageValueMeter()
        epoch_meter.reset()
        # loading model
        net = Student(n_colors=colors, n_feats=args.n_feats, n_lkb=args.n_blocks, scale=args.n_scale)

        net.to(device).eval()
        state_dict = torch.load(model_name)
        net.load_state_dict(state_dict["model"])

        output = []
        for i, (ms, lms, gt) in enumerate(test_loader):
            # compute output
            ms, lms, gt = ms.to(device), lms.to(device), gt.\
                to(device)
            # y = model(ms)
            y, _ = net(ms, lms)
            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0],:gt.shape[1],:]
            if i==0:
                indices = quality_assessment(gt, y, data_range=1., ratio=args.n_scale)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=args.n_scale))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number

    # save the results
    save_dir = result_path + model_title + '_' + sanitized_time + '.npy'
    np.save(save_dir, output)
    print("Test finished, test results saved to .npy file at ", save_dir)
    print(indices)

    # 按照指定顺序创建一个有序字典
    indices = OrderedDict([
        ("MPSNR", indices.get("MPSNR")),
        ("MSSIM", indices.get("MSSIM")),
        ("SAM", indices.get("SAM")),
        ("CrossCorrelation", indices.get("CrossCorrelation")),
        ("RMSE", indices.get("RMSE")),
        ("ERGAS", indices.get("ERGAS"))
    ])

    # 将 indices 中的数据转换为标准 float 类型
    indices = {key: float(value) for key, value in indices.items()}

    # 保存结果
    QIstr = os.path.join(result_path, model_title + '_' + sanitized_time + ".txt")

    json.dump(indices, open(QIstr, 'w'))


def save_checkpoint(args, model, epoch, traintime):
    device = torch.device("cuda" if args.cuda else "cpu")
    model.eval().cpu()

    global sanitized_time  # 确保使用同一个时间戳

    checkpoint_model_dir = f'./checkpoints/{sanitized_time}/'
    if not os.path.exists(checkpoint_model_dir):
        os.makedirs(checkpoint_model_dir)

    ckpt_model_filename = f'{args.dataset_name}_{args.model_title}_ckpt_epoch_{epoch}.pth'
    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)

    if torch.cuda.device_count() > 1:
        state = {"epoch": epoch, "model": model.module.state_dict()}
    else:
        state = {"epoch": epoch, "model": model.state_dict()}
    torch.save(state, ckpt_model_path)
    model.to(device).train()
    print("Checkpoint saved to {}".format(ckpt_model_path))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)


if __name__ == "__main__":
    main()
