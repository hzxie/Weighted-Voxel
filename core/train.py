#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Originally developed by Chris Choy <chrischoy@ai.stanford.edu>
# Updated by Haozhe Xie <cshzxie@gmail.com>

import inspect

from datetime import datetime as dt
from multiprocessing import Queue

from core.net import ReconstructionNet
from core.solver import Solver
from utils.data_io import category_model_id_pair
from utils.process_manager import make_processes, kill_processes

# Define globally accessible queues, will be used for clean exit when force interrupted
train_queue, val_queue, train_processes, val_processes = None, None, None, None

def cleanup_handle(func):
    # Cleanup the data processes before exiting the program
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            print('[INFO] %s Wait until the dataprocesses to end ...' % (dt.now()))
            kill_processes(train_queue, train_processes)
            kill_processes(val_queue, val_processes)
            raise

    return func_wrapper

@cleanup_handle
def train_net(cfg):
    net = ReconstructionNet(cfg)
    print('Network definition:')
    print(inspect.getsource(ReconstructionNet.network_definition))

    # Generate the solver
    solver = Solver(cfg, net)

    # Prefetching data processes
    #
    # Create worker and data queue for data processing. For training data, use
    # multiple processes to speed up the loading. For validation data, use 1
    # since the queue will be popped every TRAIN.NUM_VALIDATION_ITERATIONS.
    global train_queue, val_queue, train_processes, val_processes
    train_queue = Queue(cfg.TRAIN.QUEUE_SIZE)
    val_queue = Queue(cfg.TRAIN.QUEUE_SIZE)

    train_processes = make_processes(
        cfg, train_queue,
        category_model_id_pair(cfg.DIR.DATASET_TAXONOMY_FILE_PATH, cfg.DIR.DATASET_QUERY_PATH , cfg.TRAIN.DATASET_PORTION),
        cfg.TRAIN.NUM_WORKER, repeat=True)
    val_processes = make_processes(
        cfg, val_queue,
        category_model_id_pair(cfg.DIR.DATASET_TAXONOMY_FILE_PATH, cfg.DIR.DATASET_QUERY_PATH , cfg.TEST.DATASET_PORTION),
        1, repeat=True, train=False)

    # Train the network
    solver.train(train_queue, val_queue)

    # Cleanup the processes and the queue.
    kill_processes(train_queue, train_processes)
    kill_processes(val_queue, val_processes)
