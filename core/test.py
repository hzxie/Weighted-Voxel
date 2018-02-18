#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

import numpy as np

from datetime import datetime as dt
from multiprocessing import Queue

from core.net import ReconstructionNet
from core.solver import Solver
from utils.data_io import category_model_id_pair
from utils.process_manager import make_processes, kill_processes, get_while_running

def get_iou(prediction, ground_truth, threshold):
    preds_occupy = prediction >= threshold
    diff = np.sum(np.logical_xor(preds_occupy, ground_truth))
    intersection = np.sum(np.logical_and(preds_occupy, ground_truth))
    union = np.sum(np.logical_or(preds_occupy, ground_truth))

    return intersection / union

def test_net(cfg):
    # Set batch size to 1
    cfg.CONST.BATCH_SIZE = 1

    # Generate network and solver
    net = ReconstructionNet(cfg)
    print('[INFO] %s Loading network parameters from %s' % (dt.now(), cfg.CONST.WEIGHTS))
    net.load(cfg.CONST.WEIGHTS)
    
    solver = Solver(cfg, net)

    # Setup results container
    results = dict()
    results['samples']    = { t: [] for t in cfg.TEST.VOXEL_THRESH } # Result of each sample
    results['categories'] = { t: {} for t in cfg.TEST.VOXEL_THRESH } # Result of each category

    # Set up testing data process. We make only one prefetching process. The
    # process will return one batch at a time.
    queue = Queue(cfg.TRAIN.QUEUE_SIZE)
    data_pair = category_model_id_pair(cfg.DIR.DATASET_TAXONOMY_FILE_PATH, \
                                        cfg.DIR.DATASET_QUERY_PATH , \
                                        cfg.TEST.DATASET_PORTION)
    processes = make_processes(cfg, queue, data_pair, 1, repeat=False, train=False)
    n_test_data = len(processes[0].data_paths)

    # Main Test Process
    batch_idx  = 0
    categories = []

    for imgs, voxel in get_while_running(processes[0], queue):
        if batch_idx == n_test_data:
            break                       # Reach the end of the test sample + 1

        c_id = data_pair[batch_idx][0] # Category of the sample
        s_id = data_pair[batch_idx][1] # ID of the sample
        prediction, loss, activations = solver.test_output(imgs, voxel)

        # Normalize voxel to 0-1
        voxel = voxel > 0

        ious = []
        for threshold in cfg.TEST.VOXEL_THRESH:
            iou = get_iou(prediction[0, :, 1, :, :], voxel[0, :, 1, :, :], threshold)

            if not c_id in results['categories'][threshold]:
                results['categories'][threshold][c_id] = []
                if not c_id in categories:
                    categories.append(c_id)


            results['samples'][threshold].append(iou)
            results['categories'][threshold][c_id].append(iou)
            ious.append(iou)

        # Print test result of the sample
        print('[INFO] %04d/%d\tID: %s\tCategory: %s\tIoU: %s' % (batch_idx + 1, n_test_data, s_id, c_id, ious))
        batch_idx +=1

    # Print summarized results
    print('\n\n================ Test Report ================')

    print('Category ID', end='\t')
    for t in cfg.TEST.VOXEL_THRESH:
        print('%g' % t, end='\t')
    print()

    for c in categories:
        print(c, end='\t')
        for t in cfg.TEST.VOXEL_THRESH:
            print(np.mean(results['categories'][t][c]), end='\t')
        print()
        
    print('Mean', end='\t\t')
    for t in cfg.TEST.VOXEL_THRESH:
        print(np.mean(results['samples'][t]), end='\t')
    print()

    # Cleanup the processes and the queue.
    kill_processes(queue, processes)
