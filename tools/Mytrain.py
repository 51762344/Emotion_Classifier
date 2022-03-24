import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 终端运行会出现错误，故加入以下代码
import sys

sys.path.append('E:\Pycharm_Projects\DeepLearning\Classifier\Video_Emotion_Detection')

import torch
import torch.optim as optim
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
import argparse

from MyDataset import read_split_data, EmotionDataloader

from loguru import logger
from matplotlib import pylab as plt
import numpy as np
import torchvision.utils as vutils
from train_utils import train_one_epoch, evaluate, plot_data_loader_image, save_checkpoint


def make_parser():
    parser = argparse.ArgumentParser("Train parser")
    parser.add_argument('--batch_size', '-b', default=8, type=int)
    parser.add_argument('--epochs', '-e', default=5, type=int)
    parser.add_argument('--save_freq', default=3, type=int)
    parser.add_argument('--num_workers', '-nw', default=0, type=int)
    parser.add_argument('--data_dir', '-p', default=r'E:\Dataset\Classification\jaffe', type=str)
    parser.add_argument('--weights', type=str, default=None, help='initial weights path')
    parser.add_argument('--freeze', type=bool, default=False)
    parser.add_argument('--checkpoint', default=False, type=bool)
    parser.add_argument('--checkpoint_path', '-cp',
                        default=r'E:\Pycharm_Projects\DeepLearning\Classifier\Video_Emotion_Detection\checkpoint',
                        type=str)

    parser.add_argument('--tensorboard_path', '-tp',
                        default=r'E:\Pycharm_Projects\DeepLearning\Classifier\Video_Emotion_Detection\runs',
                        type=str)

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    return parser


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(device)

    tb_writer = SummaryWriter(log_dir=args.tensorboard_path)

    logger.info('Loading Data...')
    # 获取数据集图片的路径和标签
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_dir, 0.2)

    # 对数据集进行预处理并整合进DataLoader
    trainLoader = EmotionDataloader(train_images_path, train_images_label, args, train=True)
    valLoader = EmotionDataloader(val_images_path, val_images_label, args, train=False)

    # 查看数据及对应标签
    # plot_data_loader_image(trainLoader)
    # 使用torchvision的make_grid
    # images, labels = next(iter(trainLoader))
    # plt.axis("off"), plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(images, normalize=True).cpu(), (1, 2, 0)))
    # plt.show()

    logger.info('构建模型')
    model = models.resnet34(pretrained=False, num_classes=7).to(device)
    # model = models.inception_v3(pretrained=False,init_weights=True).to(device)
    if args.weights:  # 加载预训练模型
        try:
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(load_weights_dict, strict=False)
            logger.success('加载预训练模型成功')
        except:
            raise

    if args.freeze:  # 冻结权重
        for name, para in model.named_parameters():
            # 选择不需要冻结的层权重
            if "layer4" not in name:
                para.requires_grad_(False)

    logger.info('构建优化器')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    logger.info('开始训练')
    best_acc = 0.
    for epoch in range(1, args.epochs + 1):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=trainLoader,
                                                device=device,
                                                epoch=epoch
                                                )

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=valLoader,
                                     device=device,
                                     epoch=epoch)
        # 更新学习率
        scheduler.step()

        tb_writer.add_scalar('val_loss', val_loss, epoch)
        tb_writer.add_scalar('val_acc', val_acc, epoch)
        # tb_writer.add_scalar('learning_rate', val_acc, epoch)

        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.checkpoint_path + "/best_model.pth")

        # 以字典形式存储
        if args.checkpoint:
            save_checkpoint(
                {'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'opt_state_dict': optimizer.state_dict(),
                 'best_acc': best_acc},
                epoch, save_path=args.checkpoint_path, save_freq=args.save_freq)

    tb_writer.close()


if __name__ == '__main__':
    args = make_parser().parse_args()
    args.weights = \
        r'E:\Pycharm_Projects\DeepLearning\Classifier\Video_Emotion_Detection\checkpoint\My_model.pth'
    args.freeze=True
    main(args)
