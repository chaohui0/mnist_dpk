from asyncio.log import logger
import os
import typing
"""
以下为引用物源工具链包
"""
from wuyuan import spaic
from wuyuan import BrainComponent

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 19:48:52 2022

@author: editing
"""
class MNISTBrainComponent(BrainComponent):
    """
    类脑构件的公共接口。
    
    参数
    ----
    name : str
        类脑构件的名称。
    """

    def __init__(self, name: str, dt = 0.1, run_time = 3):
        super().__init__(name)
        self.dt = dt
        self.run_time = run_time
    
    def _get_preprocessing(self):
        """
        获取预处理过程。
        
        返回值
        ------
        preprocessing : typing.Callable
            预处理过程。
        """
        from component.preprocess.preprocess import preprocess
        return {'input': preprocess}

    def get_dt(self) -> float:
        """
        获取各个时间步共同的长度。
        
        返回值
        ------
            各个时间步共同的长度。
        
        注意事项
        --------
        时间步的长度是SNN模型在概念上的时间长度，而不是SNN模型的实现运行所需的真实时间长度。
        """
        return self.dt
    def get_run_time(self) -> float: 
        """
        获取SNN模型运行时间窗口的长度。
        
        返回值
        ------
            SNN模型运行时间窗口的长度。
        
        注意事项
        --------
        运行时间窗口的长度是SNN模型在概念上的时间长度，而不是SNN模型的实现运行所需的真实时间
        长度。
        """
        return self.run_time

    def save(self, path: str, save_weight: bool = True):
        """
        保存类脑构件。
        
        参数
        ----
        path : str
            类脑构件持久化存储路径。
        save_weights : bool, optional
            是否保存SNN模型的权重。
        
        注意事项
        --------
        本方法必须在BrainComponent.set_SNN或BrainComponent.load方法之后调用。
        """
        super().save(path, save_weight)
   
    def load(self, path: str, device: str, load_weight: bool = True):
        """
        载入类脑构件。
        
        参数
        ----
        path : str
            类脑构件持久化存储路径。
        load_weight : bool, optional
            是否载入SNN模型的权重。
        
        注意事项
        --------
        1. 本方法丢弃载入前的类脑构件状态。
        2. 本方法将存储各个SNN模型实现的目录中的每个目录都视为一个SNN模型实现，该目录名称即
        对应SNN模型实现的名称。
        """
        super().load(path, device, load_weight)
