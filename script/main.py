# -*- coding: utf-8 -*-

"""
Created on 2022/4/25
@project: 
@filename: app
@author: 
@contact: 
@description:手写数字识别类脑应用业务逻辑开发模板
"""

import os, sys
os.chdir(os.getcwd())
sys.path.append(os.getcwd())
import copy
import numpy as np
import cv2
import pywebio, functools, operator
import re

"""
引用物源工具链包
"""
sys.path.append("/data/workspace")
from wuyuan import Runtime
from wuyuan import DPK, PathDPKRoot

"""
引用用户自己写的类
"""
from component.mnist_brain_component import MNISTBrainComponent
"""
构造并初始化类脑构件。
用户自定义的MNISTBrainComponent在dpk里面的相对路径。
"""
bc_name_quantized = 'mnist_brain_component'
brain_component = MNISTBrainComponent(bc_name_quantized)
brain_component.load(
    path=PathDPKRoot().brain_component(bc_name_quantized).config('1'),
    device='cpu',
    load_weight=True)

"""
恢复SNN模型对象，用户拿到运行结果。
"""
model = brain_component.get_SNN()

"""
构造类脑构件运行时。
"""
run_time = 3  # 运行时间，默认为编码构件时设置的运行时长
runtime = Runtime(brain_component=brain_component)

"""
应用APP规范样例
用户自定义部分
"""
def main():

    pywebio.output.put_markdown('# Darwin 3 mnist')
    # 输入数据
    imgs = pywebio.input.file_upload("请输入图片:", 
        accept="image/*", multiple=True)
    
    # 一行输出4次结果
    row_pngs = 4
    num_correct = 0
    num_count = 0
    # 分类
    class_names = ['0', '1', '2', '3', '4',
                   '5', '6', '7', '8', '9']
    record_li = list()
    mnist = [i for i in range(10)]
    for img in imgs:
        # 获取数据
        data = img["filename"]
        if os.path.exists(img["filename"]):
            os.remove(data)
        with open(data, "wb+") as f:
            file_b = img['content']
            f.write(file_b)
        label = int(re.findall(".+label_(\d+).*", data)[0])
        image = cv2.imread(data, cv2.IMREAD_GRAYSCALE)
        data = image.reshape((1,) + image.shape)
        # 输入数据，输入数据形状为（第几个样本，此样本的数据）
        runtime.input(data, 'input')
        # 运行类脑构件
        runtime.run(time = run_time, tick_len=1)

        # 通过脉冲监控器获取计算结果。
        spikes = model.spk_l1.spk_index[0]
        sptime = model.spk_l1.spk_times[0]

        if isinstance(spikes, np.ndarray):
            spikes = spikes.tolist()
            sptime = sptime.tolist()

        ospikes = list(map(lambda x: spikes.count(x), mnist))
        monitor_res =  np.argmax(list(map(lambda x: spikes.count(x), mnist)))
        
        # 统计正确率
        if label == monitor_res:
            num_correct += 1
        num_count += 1

        # 显示
        #record_li.append([img['content'], class_names[monitor_res]+str(ospikes), str(spikes), str(sptime)])
        record_li.append([img['content'], class_names[monitor_res]])
        if len(record_li) >= row_pngs:
            show_result(copy.deepcopy(record_li))
            record_li = []
    # 显示剩余数据
    if len(record_li) != 0:
        show_result(copy.deepcopy(record_li))

    pywebio.output.put_table([[f"正确率：{(num_correct / num_count) * 100}%"]])

def show_result(record_li):
        pywebio.output.put_table([
            # ['原图片', '计算结果'],
            functools.reduce(
                operator.concat,
                list(map(lambda x: [pywebio.output.put_image(x[0]),x[1]],
                         record_li)))
        ])

"""
用户自定义程序退出需要做的事情
"""
def main_exit():
    '''用户自定义接口，APP退出进程需要做的事情，
       并由操作系统在结束应用前调用。
    '''
    print("main exit.")
    return 0

if __name__ == '__main__':
    # 获取运行服务器的端口列表
    port_list = DPK.get_port()
    app_port = port_list[0]
    # 启动服务
    pywebio.platform.tornado.start_server(main, port=int(app_port), debug=True, cdn=True, auto_open_webbrowser=False)
