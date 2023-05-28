"""Normalize features"""

import numpy as np


def normalize(features):
    """
    Desc:
        标准化
        标准化操作（standardization）是将数据按其属性（按列）减去平均值，然后再除以标准差

    parameter:
        dataSet: 数据集
    return:
        归一化后的数据集 normDataSet
        features_mean 均值
        features_deviation 标准差

    标准化公式:
        Y = (X-mean)/std
        X-mean 原点为中心左右对称
        /std 降低各维度之间的差异
        其中mean是平均值，std是标准差。。
    """
    # copy 深拷贝
    # stype函数用于array中数值类型转换
    features_normalized = np.copy(features).astype(float)

    # 计算均值
    features_mean = np.mean(features, 0)

    # 计算标准差 标准差是方差的平方根
    features_deviation = np.std(features, 0)

    # 标准化操作
    # shape读取矩阵的长度，比如shape[0]就是读取矩阵第一维度的长度
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # 防止除以0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation
