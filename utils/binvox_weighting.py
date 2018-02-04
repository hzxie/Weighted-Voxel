#!/usr/bin/python
# -*- coding: utf-8 -*-

import binvox_rw
import math
import numpy as np
import os
import sys

OMEGA = 27
VOXEL_MIN_VALUE = -53

def transform_voxel_data(voxels, n_voxs):
    weighted_voxels = np.zeros((n_voxs[0], n_voxs[1], n_voxs[2]))

    for i in range(n_voxs[0]):
        for j in range(n_voxs[1]):
            for k in range(n_voxs[2]):
                # Count zeros and ones in neighbors
                zeros_count = 0
                ones_count  = 0

                for m in range(i - 1, i + 2):
                    for n in range(j - 1, j + 2):
                        for p in range(k - 1, k + 2):
                            if m < 0 or n < 0 or p < 0 or m >= n_voxs[0] or n >= n_voxs[1] or p >= n_voxs[2]:
                                continue

                            weight = OMEGA if m == i and n == j and p == k else 1
                            if voxels[m][n][p]:
                                ones_count += weight
                            else:
                                zeros_count += weight

                # VOXEL_MIN_VALUE is used for normalizing to values with minimum value 0
                # The value in binvox cannot less than 0
                value = ones_count - zeros_count
                if value < 0:
                    value = -math.sqrt(-value) * 2
                weighted_voxels[i][j][k] = value - VOXEL_MIN_VALUE

    return weighted_voxels

def main():
    if not len(sys.argv) == 3:
        print('python binvox_weighting.py input_file_folder output_file_folder')
        sys.exit()

    input_file_folder  = sys.argv[1]
    output_file_folder = sys.argv[2]

    if not os.path.exists(input_file_folder) or not os.path.isdir(input_file_folder):
        print('[ERROR] Input folder not exists!')
        sys.exit(2)
    if not os.path.exists(output_file_folder) or not os.path.isdir(output_file_folder):
        print('[ERROR] Output folder not exists!')
        sys.exit(2)

    for category_id in os.listdir(input_file_folder):
        print('[INFO] Generating category %s' % category_id)
        category_file_folder = os.path.join(input_file_folder, category_id)

        for model_folder in os.listdir(category_file_folder):
            input_model_file_path  = os.path.join(category_file_folder, model_folder, 'model.binvox')
            output_model_folder    = os.path.join(output_file_folder, category_id, model_folder)
            output_model_file_path = os.path.join(output_model_folder, 'model.binvox')

            if not os.path.exists(input_model_file_path):
                print('[WARN] Input model file not exists in %s' % input_model_file_path)
                continue
            if not os.path.exists(output_model_folder):
                os.makedirs(output_model_folder)
            if os.path.exists(output_model_file_path):
                os.remove(output_model_file_path)

            with open(input_model_file_path, 'rb') as input_file, open(output_model_file_path, 'wb') as output_file:
                voxels = binvox_rw.read_as_3d_array(input_file)
                n_voxs = voxels.dims
                voxels.data = transform_voxel_data(voxels.data, n_voxs)
                voxels.write(output_file)

            output_file_size = os.stat(output_model_file_path).st_size
            if output_file_size == 0:
                print('[WARN] Output file size equals to zero [File=%s]' % input_model_file_path)

if __name__ == '__main__':
    main()
