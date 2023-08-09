import numpy as np

from .rescale import rescale

def preprocess(data):
    """
    为flatten生成输入数据。
    
    参数
    ----
    data :
        0到多个输入样本数据，data[i]为样本i。
    
    返回值
    ------
    preprocessed :
        上游预处理构件的输出数据，preprocessed[i]为样本i对应的数据。
    """
    return rescale(data.reshape((data.shape[0], -1)), max_value = 255.0, min_value = 0.0)
