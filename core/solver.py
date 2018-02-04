#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Originally developed by Chris Choy <chrischoy@ai.stanford.edu>
# Updated by Haozhe Xie <cshzxie@gmail.com>
#
# CHANGELOG:
# - 2018/01/20: Add dynamic learning rate adjusting policy
# - 2018/02/04: Refract the code

import numpy as np
import os
import sys
import theano
import theano.tensor as T

from datetime import datetime as dt

def max_or_nan(params):
    for param_idx, param in enumerate(params):
        # if there is a nan, max will return nan
        nan_or_max_param = np.max(np.abs(param.val.get_value()))
        print('[DEBUG] param %d: %f' % (param_idx, nan_or_max_param))
    
    return nan_or_max_param

def SGD(lr, params, grads, loss, w_decay, momentum):
    updates = []
    for param, grad in zip(params, grads):
        vel = theano.shared(param.val.get_value() * 0.)

        if param.is_bias or w_decay == 0:
            regularized_grad = grad
        else:
            regularized_grad = grad + w_decay * param.val

        param_additive = momentum * vel - lr * regularized_grad
        updates.append((vel, param_additive))
        updates.append((param.val, param.val + param_additive))

    return updates

def ADAM(lr, params, grads, loss, iteration, w_decay, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    t = iteration
    lr_t = lr * T.sqrt(1 - T.pow(beta_2, t)) / (1 - T.pow(beta_1, t))

    updates = []
    for p, g in zip(params, grads):
        m = theano.shared(p.val.get_value() * 0.)   # zero init of moment
        v = theano.shared(p.val.get_value() * 0.)   # zero init of velocity

        if p.is_bias or w_decay == 0:
            regularized_g = g
        else:
            regularized_g = g + w_decay * p.val

        m_t = (beta_1 * m) + (1 - beta_1) * regularized_g
        v_t = (beta_2 * v) + (1 - beta_2) * T.square(regularized_g)
        p_t = p.val - lr_t * m_t / (T.sqrt(v_t) + epsilon)

        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p.val, p_t))

    return updates

class Solver(object):
    def __init__(self, config, net):
        self.net            = net
        self.cfg            = config
        self.lr             = theano.shared(np.float32(1))
        self.iteration      = theano.shared(np.float32(0))  # starts from 0
        self._test          = None
        self._train_loss    = None
        self._test_output   = None

        self.compile_model(config.TRAIN.POLICY, \
                            config.TRAIN.WEIGHT_DECAY, \
                            config.TRAIN.MOMENTUM)

    def compile_model(self, policy, weight_decay, momentum):
        net         = self.net
        lr          = self.lr
        iteration   = self.iteration

        if policy == 'sgd':
            updates = SGD(lr, net.params, net.grads, net.loss, weight_decay, momentum)
        elif policy == 'adam':
            updates = ADAM(lr, net.params, net.grads, net.loss, iteration, weight_decay)
        else:
            sys.exit('[ERROR] Unimplemented optimization policy')

        self.updates = updates

    def set_lr(self, lr):
        self.lr.set_value(lr)

    @property
    def train_loss(self):
        if self._train_loss is None:
            print('[INFO] %s Compiling training function, please wait ...' % (dt.now()))
            self._train_loss = theano.function(
                [self.net.x, self.net.y], self.net.loss, updates=self.updates)
        self.iteration.set_value(self.iteration.get_value() + 1)
        
        return self._train_loss
    
    def train(self, train_queue, val_queue=None):
        ''' Given data queues, train the network '''
        # Parameter directory
        save_dir = os.path.join(self.cfg.DIR.OUT_PATH)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        training_losses = []

        # Setup start interations
        start_iter      = 0
        if self.cfg.TRAIN.RESUME_TRAIN:
            self.net.load(self.cfg.CONST.WEIGHTS)
            start_iter = self.cfg.TRAIN.INITIAL_ITERATION

        # Setup learning rates
        lr       = self.cfg.TRAIN.DEFAULT_LEARNING_RATE
        lr_steps = [int(k) for k in self.cfg.TRAIN.LEARNING_RATES.keys()]
        print('[INFO] %s Set the learning rate to %g.' % (dt.now(), lr))
        self.set_lr(lr)
       
        # Main training loop
        for train_itr in range(start_iter, self.cfg.TRAIN.NUM_ITERATION + 1):
            # Apply one gradient step
            batch_img, batch_voxel = train_queue.get()
            loss = self.train_loss(batch_img, batch_voxel)
            
            training_losses.append(loss)

            # Decrease learning rate at certain points
            if train_itr in lr_steps:
                # edict only takes string for key. Hacky way
                current_lr  = self.lr.get_value()
                expected_lr = np.float(self.cfg.TRAIN.LEARNING_RATES[str(train_itr)])

                if expected_lr < current_lr:
                    self.set_lr(expected_lr)
                print('[INFO] %s Learing rate decreased to %g: ' % (dt.now(), self.lr.get_value()))

            # Print the current loss
            if train_itr % self.cfg.TRAIN.PRINT_FREQ == 0:
                print('[INFO] %s Iter: %d Loss: %f' % (dt.now(), train_itr, loss))

            # Print test loss and params to check convergence every N iterations
            if train_itr % self.cfg.TRAIN.VALIDATION_FREQ == 0 and val_queue is not None:
                val_losses = []
                for i in range(self.cfg.TRAIN.NUM_VALIDATION_ITERATIONS):
                    batch_img, batch_voxel = val_queue.get()
                    _, val_loss, _ = self.test_output(batch_img, batch_voxel)
                    val_losses.append(val_loss)
                print('[INFO] %s Test loss: %f' % (dt.now(), np.mean(val_losses)))

            # Check that the network parameters are all valid
            if train_itr % self.cfg.TRAIN.NAN_CHECK_FREQ == 0:
                max_param = max_or_nan(self.net.params)
                if np.isnan(max_param):
                    print('[ERROR] NAN detected')
                    break

            # Save network parameters
            if train_itr % self.cfg.TRAIN.SAVE_FREQ == 0 and not train_itr == 0:
                self.save(training_losses, save_dir, train_itr)

            if loss > self.cfg.TRAIN.LOSS_LIMIT:
                print("[ERROR] Cost exceeds the threshold. Stop training ...")
                break

    def save(self, training_losses, save_dir, step):
        ''' Save the current network parameters to the save_dir and make a
        symlink to the latest param so that the training function can easily
        load the latest model'''
        save_path = os.path.join(save_dir, 'weights.%d' % (step))
        self.net.save(save_path)

        # Make a symlink for weights.npy
        symlink_path = os.path.join(save_dir, 'weights.npy')
        if os.path.lexists(symlink_path):
            os.remove(symlink_path)

        # Make a symlink to the latest network params
        os.symlink("%s.npy" % os.path.abspath(save_path), symlink_path)

        # Write the losses
        with open(os.path.join(save_dir, 'loss.%d.txt' % step), 'w') as f:
            f.write('\n'.join([str(l) for l in training_losses]))

    def test_output(self, x, y=None):
        '''
        Generate the reconstruction, loss, and activation. Evaluate loss if
        ground truth output is given. Otherwise, return reconstruction and
        activation
        '''
        # Cache the output function.
        if self._test_output is None:
            print('[INFO] %s Compiling testing function, please wait ...' % (dt.now()))
            # Lazy load the test function
            self._test_output = theano.function([self.net.x, self.net.y],
                                                [self.net.output,
                                                 self.net.loss,
                                                 *self.net.activations])
        # If the ground truth data is given, evaluate loss. O.w. feed zeros and
        # does not return the loss
        if y is None:
            n_vox = self.cfg.CONST.N_VOX
            no_loss_return = True
            y_val = np.zeros(
                        (self.cfg.CONST.BATCH_SIZE, self.n_vox, 2, self.n_vox, self.n_vox)) \
                            .astype(theano.config.floatX)
        else:
            no_loss_return = False
            y_val = y

        # Parse the result
        results     = self._test_output(x, y_val)
        prediction  = results[0]
        loss        = results[1]
        activations = results[2:]

        if no_loss_return:
            return prediction, activations
        else:
            return prediction, loss, activations
