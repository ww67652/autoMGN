import copy
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

import warnings


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def accumulate(model, dataloader, config):
    node_accumulator = Accumulator(config["model"]["node_feat_size"])
    edge_accumulator = Accumulator(config["model"]["edge_feat_size"])
    output_accumulator = Accumulator(config["model"]["output_feat_size"])
    for i, (_, _, nodes, edges, output, path) in enumerate(dataloader):
        nodes = nodes.cuda()
        edges = edges.cuda()
        output = output.cuda()

        node_accumulator.accumulate(nodes)
        edge_accumulator.accumulate(edges)
        output_accumulator.accumulate(output)

    model.node_normalizer.set_accumulated(node_accumulator)
    model.edge_normalizer.set_accumulated(edge_accumulator)
    model.output_normalizer.set_accumulated(output_accumulator)


def train(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, config):
    log = open(os.path.join(config['log_root'], 'log.txt'), 'a')

    accumulate(model, train_dataloader, config)
    for epoch in range(config['last_epoch'] + 1, config['max_epoch'] + 1):

        print('-' * 20)
        log.write('-' * 20 + '\n')

        print('Epoch: %d / %d' % (epoch, config['max_epoch']))
        log.write('Epoch: %d / %d' % (epoch, config['max_epoch']) + '\n')

        epoch_loss = 0.
        epoch_mse = 0.
        model.train()
        start = time.perf_counter()
        for i, (senders, receivers, nodes, edges, output, _) in enumerate(train_dataloader):
            senders = senders.cuda()
            receivers = receivers.cuda()
            nodes = nodes.cuda()
            edges = edges.cuda()
            output = output.cuda()

            optimizer.zero_grad()

            prediction = model(senders, receivers, nodes, edges)
            print(prediction.size())
            loss = criterion(prediction, model.output_normalizer(output))
            loss.backward()
            epoch_loss += loss.item()
            epoch_mse += torch.mean((output - model.output_normalize_inverse(prediction)) ** 2)

            optimizer.step()
            scheduler.step()

        end = time.perf_counter()
        print('Train Loss: %f, MSE: %f' % (epoch_loss / len(train_dataloader), epoch_mse / len(train_dataloader)))
        print('Train Time: %f' % (end - start))

        log.write('Train Loss: %f, MSE: %f' % (epoch_loss / len(train_dataloader), epoch_mse / len(train_dataloader)) + '\n')

        if epoch % config['eval_steps'] == 0:

            epoch_loss = 0.
            epoch_mse = 0.
            epoch_time = 0.

            model.eval()
            for i, (senders, receivers, nodes, edges, output, _) in enumerate(valid_dataloader):
                senders = senders.cuda()
                receivers = receivers.cuda()
                nodes = nodes.cuda()
                edges = edges.cuda()
                output = output.cuda()

                with torch.no_grad():
                    start = time.perf_counter()
                    prediction = model(senders, receivers, nodes, edges)
                    end = time.perf_counter()

                    loss = criterion(prediction, model.output_normalizer(output))

                    epoch_loss += loss.item()
                    epoch_mse += torch.mean((output - model.output_normalize_inverse(prediction)) ** 2)
                    epoch_time += end - start

            print('Valid Loss: %f, MSE: %f, Time Used: %f' % (
                epoch_loss / len(valid_dataloader), epoch_mse / len(valid_dataloader), epoch_time / len(valid_dataloader)))
            log.write(
                'Valid Loss: %f, MSE: %f' % (epoch_loss / len(valid_dataloader), epoch_mse / len(valid_dataloader)) + '\n')

        print('-' * 20)
        log.write('-' * 20 + '\n')

        if epoch % config['save_steps'] == 0:
            torch.save(copy.deepcopy((model.state_dict())), os.path.join(config['ckpt_root'], '%d.pkl' % epoch))

    return


if __name__ == '__main__':

    warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

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

    train_dataloader = data.DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=True)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config['decayRate'])
    train(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, config)
