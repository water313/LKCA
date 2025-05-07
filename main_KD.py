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
from teacher import Teacher
from student_kd2 import Student
from metrics import compare_mpsnr
# loss
from loss import HLoss
from KD import KD_Loss
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
    train_parser.add_argument("--epochs", type=int, default=200, help="epochs, default set to 200")
    train_parser.add_argument("--n_feats", type=int, default=128, help="n_feats, default set to 128")
    train_parser.add_argument("--n_blocks", type=int, default=16, help="n_blocks, default set to 16")
    train_parser.add_argument("--n_scale", type=int, default=4, help="n_scale, default set to 4")
    train_parser.add_argument("--dataset_name", type=str, default="Chikusei", help="dataset_name, default set to dataset_name")
    train_parser.add_argument("--model_title", type=str, default="student_kd", help="model_title, default set to model_title")
    train_parser.add_argument("--seed", type=int, default=3000, help="start seed for model")
    train_parser.add_argument("--learning_rate", type=float, default=0.002,
                              help="learning rate, default set to 2e-3")
    train_parser.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")
    train_parser.add_argument("--gpus", type=str, default="1", help="gpu ids (default: 1)")
    train_parser.add_argument("--decay_factor", type=float, default=0.66, help="kdloss衰减因子")
    train_parser.add_argument("--decay_interval", type=int, default=10, help="kdloss多少个epoch衰减一次")
    train_parser.add_argument("--init_a", type=float, default=0.01, help="kdloss占比初始值")

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

    # 初始化教师网络和学生网络
    teacher = Teacher(n_colors=colors, n_feats=args.n_feats,n_lkb=32, scale=args.n_scale)
    net = Student(n_colors=colors, n_feats=args.n_feats, n_lkb=args.n_blocks,scale=args.n_scale)

    model_title = args.model_title + '_r' + str(args.n_scale) + '_a' + str(args.init_a) + '_df' + str(args.decay_factor) + '_di' + str(args.decay_interval) + '_Feats=' + str(
        args.n_feats) + '_lkb=' + str(args.n_blocks)
    teacher_checkpoint_path = './checkpoints/teacher/Chikusei_teacher_r4_Feats=128_lkb=32_ckpt_epoch_148.pth'  # 教师网络的检查点路径
    student_checkpoint_path = './checkpoints/2025_01_13_21/iChikusei_student_kd_r4_a0.05_df0.66_di10_Feats=128_lkb=16_ckpt_epoch_147.pth'  # 学生网络的检查点路径
    args.model_title = model_title

    # 加载教师网络的检查点
    if os.path.isfile(teacher_checkpoint_path):
        print(f"===> Loading teacher checkpoint from '{teacher_checkpoint_path}'")
        teacher_checkpoint = torch.load(teacher_checkpoint_path, weights_only=True)
        teacher.load_state_dict(teacher_checkpoint['model'])
    else:
        print(f"===> No teacher checkpoint found at '{teacher_checkpoint_path}'")
        raise FileNotFoundError(f"Teacher checkpoint not found at {teacher_checkpoint_path}")

    # 冻结教师网络
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()  # 切换到评估模式
    teacher.to(device)

    # 加载学生网络的检查点（如果有）
    start_epoch = 0
    if os.path.isfile(student_checkpoint_path):
        print(f"===> Loading student checkpoint from '{student_checkpoint_path}'")
        checkpoint = torch.load(student_checkpoint_path, weights_only=True)
        start_epoch = checkpoint["epoch"]
        net.load_state_dict(checkpoint["model"])
    else:
        print(f"===> No student checkpoint found at '{student_checkpoint_path}'")

    # 多 GPU 支持
    if torch.cuda.device_count() > 1:
        print("===> Let's use", torch.cuda.device_count(), "GPUs.")
        teacher = torch.nn.DataParallel(teacher)
        net = torch.nn.DataParallel(net)

    net.to(device).train()  # 学生网络切换到训练模式

    # 打印模型结构
    print_network(teacher)
    print_network(net)

    # 定义损失函数
    h_loss = HLoss(0.5, 0.1)
    L1_loss = torch.nn.L1Loss()
    kd_loss = KD_Loss()

    print("===> Setting optimizer and logger")
    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    epoch_meter = meter.AverageValueMeter()
    writer = SummaryWriter('runs/' + model_title + '_' + sanitized_time)

    best_epoch = 0
    best_loss = float('inf')
    # 初始化参数
    a_init = args.init_a  # 初始 a 的值
    decay_factor = args.decay_factor  # 衰减因子
    decay_interval = args.decay_interval  # 每多少个 epoch 衰减一次

    print('===> Start training')
    for e in range(start_epoch, args.epochs):
        adjust_learning_rate(args.learning_rate, optimizer, e + 1)
        epoch_meter.reset()
        net.train()
        print(f"Start epoch {e + 1}, learning rate = {optimizer.param_groups[0]['lr']}")

        # 每 10 个 epoch 更新 a
        # 计算当前 epoch 对应的 a 值
        a = a_init * (decay_factor ** ((e + 1) // decay_interval))

        print(f"Epoch {e+1}: 更新参数 a 为 {a:.6f}")  # 打印更新后的值

        for iteration, (x, lms, gt) in enumerate(tqdm(train_loader, leave=False)):
            x, lms, gt = x.to(device), lms.to(device), gt.to(device)
            optimizer.zero_grad()

            # 提取教师网络和学生网络的特征
            _, teacher_features = teacher(x, lms)
            y, student_features = net(x, lms)

            # 计算损失
            loss1cos, loss1sam, loss1 = kd_loss(teacher_features, student_features)  # 知识蒸馏损失
            loss2 = h_loss(y, gt)  # 学生网络输出与 ground truth 的损失
            
            aloss = a * loss1

            loss = aloss + loss2  # 总损失
               

            epoch_meter.add(loss.item())
            loss.backward()
            optimizer.step()

            # TensorBoard 可视化
            if (iteration + log_interval) % log_interval == 0:
                print(f"===> {time.ctime()} \tEpoch[{e + 1}]({iteration + 1}/{len(train_loader)}): "
                        f"Cosine Loss: {loss1cos.item():.6f}, kdsam_Loss: {loss1sam.item():.6f}, KD Loss: {loss1.item():.6f}, Supervised Loss: {loss2.item():.6f}, Total Loss: {loss.item():.6f}, aloss: {aloss.item():.6f}")

                n_iter = e * len(train_loader) + iteration + 1
                # 分别记录知识蒸馏损失、监督损失和总损失
                writer.add_scalar('Loss/KD_Loss', loss1.item(), n_iter)
                writer.add_scalar('Loss/Supervised_Loss', loss2.item(), n_iter)
                writer.add_scalar('Loss/Total_Loss', loss.item(), n_iter)

        # 运行验证集
        #eval_loss = validate(args, eval_loader, net, L1_loss)
        #if eval_loss < best_loss:
            #best_loss = eval_loss
            #best_epoch = e + 1
            # save_checkpoint(args, net, e + 1, sanitized_time)
            # print(f"Saved best model at epoch {e + 1} with loss {best_loss:.6f}")

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
        #print("===> {}\tEpoch evaluation Complete: Avg. Loss: {:.6f}, best_epoch: {}, best_loss: {:.6f}, MPSNR: {:.4f}".format( time.ctime(), eval_loss, best_epoch, best_loss, mpsnr))

        # TensorBoard 可视化
        #writer.add_scalar('scalar/avg_epoch_loss', epoch_meter.value()[0], e + 1)
        #writer.add_scalar('scalar/avg_validation_loss', eval_loss, e + 1)
        writer.add_scalar('scalar/epoch', e+1, e+1)
        writer.add_scalar('scalar/best_epoch', best_epoch, e+1)
        writer.add_scalar('scalar/mpsnr', mpsnr, e + 1)
        writer.add_scalar('scalar/best_mpsnr', best_mpsnr, e + 1)

        # save model weights at checkpoints every 5 epochs
        if (e + 1) % 1 == 0:
            save_checkpoint(args, net, e+1, sanitized_time)


def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp


def adjust_learning_rate(start_lr, optimizer, epoch):
    lr = start_lr * (0.1 ** (epoch // 100))  

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
