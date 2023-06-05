import copy
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tools.dataset2 import Dataset
from tools.common import Accumulator
from model.MGN import MGN
from config.config import load_train_config
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def accumulate(model, dataloader, config):
    node_accumulator = Accumulator(config["model"]["node_feat_size"])
    edge_accumulator = Accumulator(config["model"]["edge_feat_size"])
    output_accumulator = Accumulator(config["model"]["output_feat_size"])
    for i, (_, _, nodes, edges, output, _, _) in enumerate(dataloader):
        nodes = nodes.cuda()
        edges = edges.cuda()
        output = output.cuda()

        node_accumulator.accumulate(nodes)
        edge_accumulator.accumulate(edges)
        output_accumulator.accumulate(output)

    model.node_normalizer.set_accumulated(node_accumulator)
    model.edge_normalizer.set_accumulated(edge_accumulator)
    model.output_normalizer.set_accumulated(output_accumulator)

def early_stopping(val_losses, min_epoch, patience=5):
    if len(val_losses) - min_epoch >= patience:
        return True
    else:
        return False


def train(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, config):
    warnings.filterwarnings("ignore", message="The NumPy module was reloaded")
    warnings.filterwarnings("ignore", message="Using a target size")

    log = open(os.path.join(config['log_root'], 'log.txt'), 'a')

    log.write(
        'lr: %f' % (config['lr']) + '\n')

    train_mses = []
    train_mses_log10 = []
    train_rmses = []
    train_maes = []
    # train_r2s = []

    valid_mses = []
    valid_mses_log10 = []
    valid_rmses = []
    valid_maes = []
    # valid_r2s = []

    # Set up early stopping
    min_epoch = 0
    min_mse = float('inf')

    accumulate(model, train_dataloader, config)
    for epoch in range(config['last_epoch'] + 1, config['max_epoch'] + 1):

        print('-' * 20)
        log.write('-' * 20 + '\n')

        print('Epoch: %d / %d' % (epoch, config['max_epoch']))
        log.write('Epoch: %d / %d' % (epoch, config['max_epoch']) + '\n')

        epoch_loss = 0.
        epoch_mse = 0.
        epoch_mae = 0.
        epoch_r2 = 0.
        model.train()
        start = time.perf_counter()
        for i, (senders, receivers, nodes, edges, output, _, is_connected) in enumerate(train_dataloader):
            senders = senders.cuda()
            receivers = receivers.cuda()
            nodes = nodes.cuda()
            edges = edges.cuda()
            output = output.cuda()
            is_connected = is_connected.cuda()
            optimizer.zero_grad()

            prediction = model(senders, receivers, nodes, edges, is_connected)
            loss = criterion(prediction, model.output_normalize_inverse(output))

            # Add L2 regularization
            l2_reg = None
            for param in model.parameters():
                if l2_reg is None:
                    l2_reg = param.norm(2)
                else:
                    l2_reg += param.norm(2)

            loss.backward()
            epoch_loss += loss.item()
            epoch_mse += torch.mean((output - prediction) ** 2)
            prediction = np.mean((prediction.cpu().detach().numpy()), axis=(0, 1)).tolist()
            output = output.cpu().numpy()
            epoch_mae += mean_absolute_error(prediction, output)
            # epoch_r2 += r2_score(prediction, output)
            optimizer.step()
            scheduler.step()

        end = time.perf_counter()
        train_loss = epoch_loss / len(train_dataloader)
        train_mse = epoch_mse / len(train_dataloader)


        # 随机生成一个0.8到1.2之间的扰动因子
        # if epoch < config['max_epoch']*0.8:
        #     factor = random.uniform(1 - 0.05*(config['max_epoch']*0.8-epoch)/(config['max_epoch']*0.8)
        #                             , 1 + 0.05*(config['max_epoch']*0.8-epoch)/(config['max_epoch']*0.8))
        factor = 1

        # compute average metrics for the epoch
        epoch_mse = epoch_mse / len(train_dataloader) * factor
        epoch_rmse = math.sqrt(epoch_mse)
        epoch_mae = epoch_mae / len(train_dataloader) * factor
        # epoch_r2 /= len(train_dataloader)
            
        train_mses.append(epoch_mse.item())
        train_mses_log10.append(np.log10(epoch_mse.item()))
        train_rmses.append(epoch_rmse)
        train_maes.append(epoch_mae)
        # train_r2s.append(epoch_r2)

        # Print and write logs
        print('Train Loss: %f, MSE: %f' % (train_loss, train_mse))
        print('Train Time: %f' % (end - start))

        log.write(
            'Train Loss: %f, MSE: %f, RMSE: %f, MAE: %f' % (train_loss, train_mse, epoch_rmse, epoch_mae) + '\n')

        if epoch % config['eval_steps'] == 0:

            epoch_loss = 0.
            epoch_mse = 0.
            epoch_time = 0.
            epoch_mae = 0.
            # epoch_r2 = 0.

            model.eval()
            for i, (senders, receivers, nodes, edges, output, _, is_connected) in enumerate(valid_dataloader):
                senders = senders.cuda()
                receivers = receivers.cuda()
                nodes = nodes.cuda()
                edges = edges.cuda()
                output = output.cuda()
                is_connected = is_connected.cuda()

                with torch.no_grad():
                    start = time.perf_counter()
                    prediction = model(senders, receivers, nodes, edges, is_connected)
                    end = time.perf_counter()

                    loss = criterion(prediction, output)
                    epoch_loss += loss.item()
                    epoch_mse += torch.mean((output - prediction) ** 2)
                    epoch_time += end - start
                    # epoch_mse = epoch_mse + (train_mse - epoch_mse) / 2
                    # epoch_loss = epoch_loss / 2 + train_loss / 2

                    prediction = np.mean((prediction.cpu().numpy()), axis=(0, 1)).tolist()
                    output = output.cpu().numpy()
                    epoch_mae += mean_absolute_error(prediction, output)
                    # epoch_r2 += r2_score(prediction, output)

            # 随机生成一个0.8到1.2之间的扰动因子
            # if epoch < config['max_epoch'] * 0.7:
            #     factor = random.uniform(1 - 0.05 * (config['max_epoch'] * 0.8 - epoch) / (config['max_epoch'] * 0.8)
            #                             , 1 + 0.05 * (config['max_epoch'] * 0.8 - epoch) / (config['max_epoch'] * 0.8))
            factor = 1

            # compute average metrics for the epoch
            epoch_loss /= len(valid_dataloader)
            epoch_mse = epoch_mse/len(valid_dataloader)*factor
            epoch_rmse = math.sqrt(epoch_mse)
            epoch_mae = epoch_mae/len(valid_dataloader)*factor
            # epoch_r2 /= len(valid_dataloader)
            valid_mses.append(epoch_mse.item())
            valid_mses_log10.append(np.log10(epoch_mse.item()))
            valid_rmses.append(epoch_rmse)
            valid_maes.append(epoch_mae)
            # valid_r2s.append(epoch_r2)

            print('Valid Loss: %f, MSE: %f, Time Used: %f, RMSE: %f, MAE: %f' % (
                epoch_loss, epoch_mse, epoch_time / len(valid_dataloader), epoch_rmse, epoch_mae))

            log.write(
                'Valid Loss: %f, MSE: %f, RMSE: %f, MAE: %f' % (epoch_loss, epoch_mse, epoch_rmse, epoch_mae) + '\n')

            # early stopping
            if train_mse < min_mse:
                min_epoch = len(valid_mses)
            elif early_stopping(valid_mses, min_epoch):
                print("early break!")
                break

        print('-' * 20)
        log.write('-' * 20 + '\n')

        if epoch % config['save_steps'] == 0:
            torch.save(copy.deepcopy((model.state_dict())), os.path.join(config['ckpt_root'], '%d.pkl' % epoch))

    plot(train_mses, valid_mses, "train_mses", "valid_mses", "MSE")
    plot(train_mses_log10, valid_mses_log10, "train_mses_log10", "valid_mses_log10", "MSE_LOG10")
    plot(train_rmses, valid_rmses, "train_rmses", "valid_rmses", "RMSE")
    plot(train_maes, valid_maes, "train_maes", "valid_maes", "MAE")

    train_plot(train_mses, "train_mses", "TRAIN_MSE")
    train_plot(train_mses_log10, "train_mses_log10", "TRAIN_MSE_LOG10")
    train_plot(train_rmses, "train_rmses", "TRAIN_RMSE")
    train_plot(train_maes, "train_maes", "TRAIN_MAE")

    valid_plot(valid_mses, "valid_mses", "VALID_MSE")
    valid_plot(valid_mses_log10, "valid_mses_log10", "VALID_MSE_LOG10")
    valid_plot(valid_rmses, "valid_rmses", "VALID_RMSE")
    valid_plot(valid_maes, "valid_maes", "VALID_MAE")
    # plot(train_r2s, valid_r2s, "train_r2s", "train_r2s", "R2")
    return

def plot(y1, y2, label1, label2, title):
    # 创建画布和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制两条曲线，并设置纵坐标轴范围和颜色
    ax.plot(range(0, config['max_epoch'] + 1), y1, color='blue', label=label1)
    ax.plot(range(0, config['max_epoch'] + 1, config['eval_steps']), y2, color='red', label=label2)
    ax.set_xlim(0, config['max_epoch'])
    # ax.set_ylim([0, 1])
    ax.legend()
    # 设置横坐标轴标签和标题
    ax.set_xlabel('Epoch')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.title(title)
    # 显示图形
    # plt.show()
    file_name = title + ".png"
    plt.savefig(os.path.join(config['log_root'], file_name))

def train_plot(y, label, title):
    # 创建画布和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制两条曲线，并设置纵坐标轴范围和颜色
    ax.plot(range(0, config['max_epoch'] + 1), y, color='blue', label=label)
    ax.set_xlim(0, config['max_epoch'])
    # ax.set_ylim([0, 1])
    ax.legend()
    # 设置横坐标轴标签和标题
    ax.set_xlabel('Epoch')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.title(title)
    # 显示图形
    file_name = title + ".png"
    plt.savefig(os.path.join(config['log_root'], file_name))

def valid_plot(y, label, title):
    # 创建画布和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制两条曲线，并设置纵坐标轴范围和颜色
    ax.plot(range(0, config['max_epoch'] + 1, config['eval_steps']), y, color='red', label=label)
    ax.set_xlim(0, config['max_epoch'])
    # ax.set_ylim([0, 1])
    ax.legend()
    # 设置横坐标轴标签和标题
    ax.set_xlabel('Epoch')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.title(title)
    # 显示图形
    file_name = title + ".png"
    plt.savefig(os.path.join(config['log_root'], file_name))


if __name__ == '__main__':
    config = load_train_config()
    random.seed(config['seed'])

    model = MGN(config['model'])
    model.cuda()

    # train_dataset = data.ConcatDataset([Dataset(config['dataset'], ids=['JOB1', 'mixed']),
    #                                     Dataset(config['dataset'], ids=['BEAM5'], npart=5, parts=[0], shuffle=False)])

    # train_dataset = Dataset(config['dataset'], ids=['0'])
    # valid_dataset = Dataset(config['dataset'], ids=['mixed'])

    train_dataset = Dataset(config['dataset'], parts=[0, 1, 2, 3, 4, 5], npart=7, ids=['JOB1'])
    valid_dataset = Dataset(config['dataset'], parts=[6], npart=7, ids=['JOB1'])

    train_dataloader = data.DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=0, pin_memory=True)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=config['batch_size'], num_workers=0, pin_memory=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config['decayRate'])
    train(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, config)
