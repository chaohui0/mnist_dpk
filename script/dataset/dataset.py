import numpy as np
import os
import struct
import typing

"""
工具应该根据公用接口自动为生成代码框架，由用户填充代码来实现功能。
"""

class Dataset:
    """
    数据子集。
    
    参数
    ----
    is_train : bool
        所属实例表示训练集（True）或测试集（False）。
    path : str
        数据集路径。
    kwargs :
        其他参数，如数据集的文件路径。
    """
    def __init__(self, is_train:bool = True, **kwargs):
        root = str(kwargs.get('path', '.'))
        if bool(is_train):
            file_examples, file_labels = Dataset._FILES['train']
        else:
            file_examples, file_labels = Dataset._FILES['test']
        self._examples = Dataset._load_example_file(os.path.join(root, file_examples))
        self._labels = Dataset._load_label_file(os.path.join(root, file_labels))
        if self._examples.shape[0] != self._labels.shape[0]:
            raise Exception('The number of examples must be equal to the number of labels.')
    
    def __getitem__(self, indices:typing.Iterable[int]):
        """
        获取若干给定序号对应的样本和其他数据（如标签）。
        
        参数
        ----
        indices :
            序号。
        
        返回值
        ------
        examples : numpy.ndarray
            手写数字图像， *examples[i]* 是 *indices[i]* 对应的手写数字图像。
        labels : numpy.ndarray
            数字标签， *labels[i]* 是 *indices[i]* 对应的数字标签。
        """
        return self._examples[indices, ...], self._labels[indices, ...]
    
    def __len__(self):
        """
        获取样本数量。
        
        返回值
        ------
        num_examples : int
            样本数量。
        """
        return self._examples.shape[0]
    
    _FILES = {
        'train': ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
        'test': ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
    }
    _OFFSET_SIZE = 4
    _LENGTH_SIZE = 4
    _OFFSET_EXAMPLE = 8
    _WIDTH = 28
    _HEIGHT = 28
    
    @staticmethod
    def _load_example_file(path:str):
        print('path:', path)
        length_example = Dataset._WIDTH * Dataset._HEIGHT
        with open(str(path), 'rb') as file:
            file.seek(Dataset._OFFSET_SIZE, os.SEEK_CUR)
            size = struct.unpack('>I', file.read(Dataset._LENGTH_SIZE))[0]
            file.seek(Dataset._OFFSET_EXAMPLE, os.SEEK_CUR)
            examples = \
                np.array(
                    struct.unpack(f'>{size * length_example}B', file.read(size * length_example)),
                    dtype = np.float32)
            return examples.reshape((size, Dataset._WIDTH, Dataset._HEIGHT))
    
    @staticmethod
    def _load_label_file(path:str):
        print(path)
        with open(str(path), 'rb') as file:
            file.seek(Dataset._OFFSET_SIZE, os.SEEK_CUR)
            size = struct.unpack('>I', file.read(Dataset._LENGTH_SIZE))[0]
            return np.array(struct.unpack(f'>{size}B', file.read(size)), dtype = np.int32)

