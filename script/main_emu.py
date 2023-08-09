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

print(os.getcwd())
# script作为根目录
os.chdir(os.path.join(os.getcwd(), '.'))
# script搜索路径
sys.path.insert(0, os.getcwd())
# sys.path.insert(0, os.path.join(os.getcwd(), '../../../'))
sys.path.append("/data/workspace")
"""
引用物源工具链包
"""
from wuyuan import Runtime
from wuyuan import PathDPKRoot

"""
引用用户自己写的类
"""
from component.mnist_brain_component import MNISTBrainComponent

from wuyuan.runtime.runtime_emulator import RuntimeEmulator
Runtime.register( RuntimeEmulator)
# from wuyuan.runtime.runtime_os import RuntimeOS
# Runtime.register('runtime_os', RuntimeOS)
#from wuyuan.runtime.runtime_spaic import RuntimeSPAIC
#Runtime.register('runtime_spaic', RuntimeSPAIC)

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
def print_net_param(net, bit_width = 8):
    print("Starting to weight_quantify")
    for conn in net.get_connections():
        v_th = conn.post.parameters.get('v_th', 1.0)
        print("print_net_param:",v_th)
print_net_param(model)

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
    data_path = os.path.join(os.getcwd(), "test_images")
    files = os.listdir(data_path)
    files.sort()
    batch_size = 1
    # print(files)
    for index in range(0, len(files)-batch_size, batch_size):
        batch_data = []
        labels = []
        for step in range(batch_size):
            label = int(re.findall(".+label_(\d+).*", files[index+step])[0])
            labels.append(label)
            file_path = os.path.join(data_path, files[index+step])
            # print(file_path)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            batch_data.append(np.array(image).flatten())
        # data = image.reshape((1,) + image.shape)
        # 输入数据
        batch_data = np.array(batch_data)
        # runtime.reset()
        runtime.input(batch_data, 'input')
        # 运行类脑构件
        runtime.run(run_time)
        # runtime.run_continue(time=1)
        # runtime.run_continue(time=1)
        # runtime.run_continue(time=1)
        """
        获取计算结果。
        """
        for index in range(batch_size):
            spikes = model.spk_l1.spk_index[0]
            sptime = model.spk_l1.spk_times[0]
            print("mon_V.value:", model.mon_V.values)
            print("mon_V.times:", model.mon_V.times)

            # if isinstance(spikes, np.ndarray):
            #     spikes = spikes.tolist()
            #     sptime = sptime.tolist()
            #     print("spikes:", spikes)
            #     ospikes = list(map(lambda x: spikes.count(x), mnist))
            #     print("ospikes:", ospikes)

            # merge_list = []
            # for item_i, item_t in zip(spikes, sptime):
            #     merge_list.append((item_i,item_t))
            # print(merge_list)
            # merge_list = sorted(merge_list, key=lambda x: x[1], reverse=False)
            # spikes=list(map(lambda x:x[0], merge_list))
            # print("sort spikes:", spikes)
            # sptimes=list(map(lambda x:round(x[1],1), merge_list))
            # print("sort sptimes:", sptimes)

            # input_m = model.mon_I
            # print("input_m spk_index:", input_m.spk_index)
            # print("input_m spk_times:", input_m.spk_times)
            print("records size :",model.output.records.size(), model.output.records)
            output_spikes = model.output.predict[index]
            print(output_spikes)
            if isinstance(output_spikes, torch.Tensor):
                output_spikes = output_spikes.detach().numpy()
                print("output_spikes:", output_spikes)
            output_res = np.argmax(output_spikes)
            label = labels[index]
            print(f'label:{label}, output_res:{output_res}')
            num_count += 1
            if label == output_res:
                num_correct += 1
      
        break

    print(f"accuracy:{float(num_correct)/num_count}")

if __name__ == '__main__':
    main()
