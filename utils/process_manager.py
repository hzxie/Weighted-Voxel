#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Originally developed by Chris Choy <chrischoy@ai.stanford.edu>
# Updated by Haozhe Xie <cshzxie@gmail.com>
# 
# CHANGELOG:
# - 2018/01/11: Adapted for Weight Voxel data

import numpy as np
import sys
import theano
import traceback

from datetime import datetime as dt
from multiprocessing import Process, Event
from PIL import Image
from time import sleep

from utils.data_augmentation import preprocess_image
from utils.data_io import get_voxel_file, get_rendering_file
from utils.binvox_rw import read_as_3d_array

def print_error(func):
    '''Flush out error messages. Mainly used for debugging separate processes'''
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            traceback.print_exception(*sys.exc_info())
            sys.stdout.flush()

    return func_wrapper

class DataProcess(Process):
    def __init__(self, config, data_queue, data_paths, repeat=True):
        '''
        data_queue : Multiprocessing queue
        data_paths : list of data and label pair used to load data
        repeat : if set True, return data until exit is set
        '''
        super(DataProcess, self).__init__()
        # Queue to transfer the loaded mini batches
        self.data_queue = data_queue
        self.data_paths = data_paths
        self.num_data = len(data_paths)
        self.repeat = repeat

        # Tuple of data shape
        self.batch_size = config.CONST.BATCH_SIZE
        self.exit = Event()
        self.shuffle_db_inds()

    def shuffle_db_inds(self):
        # Randomly permute the training roidb
        if self.repeat:
            self.perm = np.random.permutation(np.arange(self.num_data))
        else:
            self.perm = np.arange(self.num_data)
        self.cur = 0

    def get_next_minibatch(self):
        if (self.cur + self.batch_size) >= self.num_data and self.repeat:
            self.shuffle_db_inds()

        db_inds = self.perm[self.cur:min(self.cur + self.batch_size, self.num_data)]
        self.cur += self.batch_size
        return db_inds

    def shutdown(self):
        self.exit.set()

    @print_error
    def run(self):
        iteration = 0
        # Run the loop until exit flag is set
        while not self.exit.is_set() and self.cur <= self.num_data:
            # Ensure that the network sees (almost) all data per epoch
            db_inds = self.get_next_minibatch()

            data_list = []
            label_list = []
            for batch_id, db_ind in enumerate(db_inds):
                datum = self.load_datum(self.data_paths[db_ind])
                label = self.load_label(self.data_paths[db_ind])

                data_list.append(datum)
                label_list.append(label)

            batch_data = np.array(data_list).astype(np.float32)
            batch_label = np.array(label_list).astype(np.float32)

            # The following will wait until the queue frees
            self.data_queue.put((batch_data, batch_label), block=True)
            iteration += 1

    def load_datum(self, path):
        pass

    def load_label(self, path):
        pass

class ReconstructionDataProcess(DataProcess):
    def __init__(self, config, data_queue, category_model_pair, background_imgs=[], repeat=True,
                 train=True):
        self.cfg = config
        self.repeat = repeat
        self.train = train
        self.background_imgs = background_imgs
        super(ReconstructionDataProcess, self).__init__(
                config, data_queue, category_model_pair, repeat=repeat)

    @print_error
    def run(self):
        # set up constants
        img_h = self.cfg.CONST.IMG_W
        img_w = self.cfg.CONST.IMG_H
        n_vox = self.cfg.CONST.N_VOX
        n_views = self.cfg.CONST.N_VIEWS

        while not self.exit.is_set() and self.cur <= self.num_data:
            # To insure that the network sees (almost) all images per epoch
            db_inds = self.get_next_minibatch()
            # We will sample # views
            if self.cfg.TRAIN.RANDOM_NUM_VIEWS:
                curr_n_views = np.random.randint(n_views) + 1
            else:
                curr_n_views = n_views

            # This will be fed into the queue. create new batch everytime
            batch_img = np.zeros(
                (curr_n_views, self.batch_size, 3, img_h, img_w), dtype=theano.config.floatX)
            batch_voxel = np.zeros(
                (self.batch_size, n_vox, 2, n_vox, n_vox), dtype=theano.config.floatX)

            # load each data instance
            for batch_id, db_ind in enumerate(db_inds):
                category, model_id = self.data_paths[db_ind]
                image_ids = np.random.choice(self.cfg.TRAIN.NUM_RENDERING, curr_n_views)

                # load multi view images
                for view_id, image_id in enumerate(image_ids):
                    img = self.load_img(category, model_id, image_id)
                    # channel, height, width
                    batch_img[view_id, batch_id, :, :, :] = \
                        img.transpose((2, 0, 1)).astype(theano.config.floatX)

                voxel = self.load_label(category, model_id)
                voxel_data = voxel.data
                # The batch_voxel is now filled with integers
                batch_voxel[batch_id, :, 0, :, :] = -voxel_data
                batch_voxel[batch_id, :, 1, :, :] = voxel_data

            # The following will wait until the queue frees
            self.data_queue.put((batch_img, batch_voxel), block=True)

        print('[INFO] %s Done. Exiting ReconstructionDataProcess ...' % (dt.now()))

    def load_img(self, category, model_id, image_id):
        image_fn = get_rendering_file(self.cfg.DIR.RENDERING_PATH, category, model_id, image_id)
        img = Image.open(image_fn)

        return preprocess_image(img, self.cfg, self.train)

    def load_label(self, category, model_id):
        voxel_fn = get_voxel_file(self.cfg.DIR.VOXEL_PATH, category, model_id)
        with open(voxel_fn, 'rb') as f:
            voxel = read_as_3d_array(f)
            # Set the original value to [MIN_VOXEL_VALUE, -MIN_VOXEL_VALUE] 
            # instead of [0, -MIN_VOXEL_VALUE * 2)
            voxel.data  = voxel.data.astype(int)
            voxel.data += self.cfg.CONST.MIN_VOXEL_VALUE

        return voxel

def kill_processes(queue, processes):
    print('[INFO] %s Send signal to processes' % (dt.now()))
    for p in processes:
        p.shutdown()

    print('[INFO] %s Empty queue' % (dt.now()))
    while not queue.empty():
        sleep(0.5)
        queue.get(False)

    print('[INFO] %s kill processes' % (dt.now()))
    for p in processes:
        p.terminate()

def make_processes(cfg, queue, data_paths, num_workers, repeat=True, train=True):
    '''Make a set of data processes for parallel data loading.'''
    processes = []
    for i in range(num_workers):
        process = ReconstructionDataProcess(cfg, queue, data_paths, repeat=repeat, train=train)
        process.start()
        processes.append(process)
    
    return processes

def get_while_running(data_process, data_queue, sleep_time=0):
    while True:
        sleep(sleep_time)
        try:
            batch_data, batch_label = data_queue.get_nowait()
        except queue.Empty:
            if not data_process.is_alive():
                break
            else:
                continue
        
        yield batch_data, batch_label

