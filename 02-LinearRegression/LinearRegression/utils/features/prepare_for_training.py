"""Prepares the dataset for training"""

import numpy as np
from .normalize import normalize
from .generate_sinusoids import generate_sinusoids
from .generate_polynomials import generate_polynomials


def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    """
        预处理
    Args:
        data (): 数据
        polynomial_degree ():
        sinusoid_degree ():
        normalize_data ():

    Returns:

    """
    # 计算样本总数
    num_examples = data.shape[0]

    #拷贝数据
    data_processed = np.copy(data)

    # 预处理
    #均值
    features_mean = 0
    #特征偏差
    features_deviation = 0

    # 规范化数据
    data_normalized = data_processed

    # 是否需要做标转化操作
    if normalize_data:
        # 返回值
        # normDataSet  归一化后的数据集
        # features_mean  均值
        # features_deviation  标准差
        (
            data_normalized,
            features_mean,
            features_deviation
        ) = normalize(data_processed)

        data_processed = data_normalized

    # 特征变换sinusoidal
    if sinusoid_degree > 0:
        sinusoids = generate_sinusoids(data_normalized, sinusoid_degree)
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)

    # 特征变换polynomial
    if polynomial_degree > 0:
        polynomials = generate_polynomials(data_normalized, polynomial_degree, normalize_data)
        data_processed = np.concatenate((data_processed, polynomials), axis=1)

    # 加一列1
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))

    return data_processed, features_mean, features_deviation
