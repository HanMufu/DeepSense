#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/3/27 19:18
# @Author  : qin yuxin
# @File    : reshape_file.py
# @Software: PyCharm


import os
import numpy as np

dataDir = './testData'
# trainDir = os.path.join(dataDir, 'train')
# evalDir = os.path.join(dataDir, 'eval')  # 跑真实数据的时候要用 现在只选了50个 所以没有分train和eval


def generate_sample_weight_array(T):
    T = int(T)
    return [1, ] * T + [0, ] * (20 - T)


def generate_arrays_from_file(sample_length_dict, batchsize=16):
    path = dataDir
    while True:
        cnt = 0
        X = []
        Y = []
        weight = []
        sample_list = os.listdir(path)
        for sample_name in sample_list:
            with open(os.path.join(path, sample_name)) as file_in:
                cnt += 1
                sample = file_in.readline()[:-1].split(',')
                # create numpy arrays of input data
                # and labels, from each line in the file
                x, y = get_feature_and_label(sample)
                X.append(x)
                Y.append(y)
                weight.append(generate_sample_weight_array(sample_length_dict[sample_name]))
                if cnt == batchsize:
                    cnt = 0
                    X = np.asarray(X)
                    X = X.transpose((1, 0, 2, 3))
                    X = np.reshape(X, newshape=[40, -1, 1, 2, 12])
                    X = list(X)
                    Y = np.asarray(Y)
                    weight = np.asarray(weight)
                    yield (X, {'output_FC': Y}, weight)
                    X = []
                    Y = []
                    weight = []


def get_feature_and_label(sample):
    speed_x = np.asarray(sample[:240]).reshape(20, 12)
    speed_y = np.asarray(sample[240:480]).reshape(20, 12)
    acc_x = np.asarray(sample[480:720]).reshape(20, 12)
    acc_y = np.asarray(sample[720:960]).reshape(20, 12)
    displace = np.asarray(sample[960:]).reshape(20, 2)

    res_speed = []
    res_acc = []
    res_t = []
    for i in range(20):
        res_speed.append(np.asarray([speed_x[i], speed_y[i]]))
        res_acc.append(np.asarray([acc_x[i], acc_y[i]]))

    for i in range(len(res_speed)):
        res_t.append(res_speed[i])
        res_t.append(res_acc[i])

    return res_t, displace


if __name__ == '__main__':
    features, label = get_feature_and_label()
    # print(features.shape)
    # print(label.shape)
