# -*- coding: utf-8 -*-
"""
Created on 2022/4/25
@project: 
@filename: app
@author: wangfengjuan
@contact: wangfj@zhejianglab.com
@description:用户类脑应用运行流程
"""
import os, sys
import cv2
import time
import numpy as np
import torch

# script作为根目录
os.chdir(os.path.join(os.getcwd(), 'test/mnist_full/mnist_full_dpk/script'))
# script搜索路径
sys.path.insert(0, os.getcwd())

"""
引用物源工具链包
"""
from wuyuan import Runtime
from wuyuan import PathDPKRoot
# from wuyuan.runtime.runtime_emulator import RuntimeEmulator
# # from .runtime_emulator import RuntimeEmulator
# Runtime.register('runtime_emu', RuntimeEmulator)
"""
引用用户自己写的类
"""
from component.mnist_brain_component import MNISTBrainComponent
"""
构造并初始化类脑构件。
用户自定义的MNISTBrainComponent在dpk里面的相对路径。
"""
dt = 0.1
run_time = 3
bc_name_quantized = 'mnist_brain_component'
mnist_brain_component = MNISTBrainComponent(bc_name_quantized, dt=dt,
    run_time=run_time)
mnist_brain_component.load(
    path = PathDPKRoot().brain_component(bc_name_quantized).config('1'),
    device = 'cpu',
    load_weight = True)
model = mnist_brain_component.get_SNN()

"""
构造类脑构件运行时。
"""
runtime = Runtime(brain_component=mnist_brain_component)

"""
应用APP规范样例
用户自定义部分
"""
def main():
    import re
    num_correct = 0
    num_count = 0
    time_start = time.time()
    mnist = [i for i in range(10)]
    data_path = os.path.join(os.getcwd(), "../../../dataset/minst/test_images")
    files = os.listdir(data_path)
    batch_size = 3
    
    for index in range(0, len(files)-batch_size, batch_size):
        batch_data = []
        labels = []
        for step in range(batch_size):
            label = int(re.findall(".+label_(\d+).*", files[index+step])[0])
            labels.append(label)
            file_path = os.path.join(data_path, files[index+step])
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            batch_data.append(np.array(image))
        # data = image.reshape((1,) + image.shape)
        # 输入数据
        batch_data = np.array(batch_data)
        # runtime.reset()
        runtime.input(batch_data)
        # 运行类脑构件
        runtime.run(run_time)
        # runtime.run_continue(time=1)
        # runtime.run_continue(time=1)
        # runtime.run_continue(time=1)
        """
        获取计算结果。
        """
        for index in range(batch_size):
            output_spikes = model.output.predict[index]
            if isinstance(output_spikes, torch.Tensor):
                output_spikes = output_spikes.detach().numpy()
            output_res = np.argmax(output_spikes)
            spikes = model.monitor2.spk_index[index]
            # spikes_times = model.monitor2.spk_times[0]
            if isinstance(spikes, np.ndarray):
                spikes = spikes.tolist()
            mnist_count = list(map(lambda x:spikes.count(x), mnist))
            cal_res = np.argmax(mnist_count)
            label = labels[index]
            print(f'label:{label}, monitor:{cal_res}, output_res:{output_res}')
            num_count += 1
            if label == cal_res:
                num_correct += 1
        
        # runtime.reset()
        # break
    time_end = time.time()
    cost_time = time_end - time_start

    print(f"cost time {cost_time} s, count:{num_count}, per: {cost_time / num_count}.")
    print(f"correct : {(num_correct / num_count) * 100}%")

if __name__ == '__main__':
    main()
