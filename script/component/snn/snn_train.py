# -*- coding: utf-8 -*-
import os,sys
import torch
import torch.nn.functional as F
from pathlib import Path

# script作为根目录
os.chdir(os.path.join(os.getcwd(), '.'))
# script搜索路径
sys.path.insert(0, os.getcwd())
sys.path.append("/data/workspace")

from wuyuan import spaic
from wuyuan import PathDPKRoot
from component.mnist_brain_component import MNISTBrainComponent

# 创建类脑构件实例
bc_name = 'mnist_brain_component'
brain_component = MNISTBrainComponent(bc_name)


class SNNModel(spaic.Network):
     def __init__(self, time):
        super(SNNModel, self).__init__()
        # 为输入层创建输入组。
        self.input = spaic.Encoder(shape=(784,), coding_method='uniform')
        # 为隐藏层创建神经元组。
        self.layer1 = spaic.NeuronGroup(shape=(10,), model='if')
        # 为输出层创建神经元组。
        # 创建从输入组到隐藏层神经元组的连接。
        self.connection1 = spaic.Connection(self.input,  self.layer1, link_type = 'full')

        # # 为输出层神经元组添加膜电位监视器。
        self.mon_V = spaic.StateMonitor(self.layer1, 'V')
        # self.mon_I = spaic.SpikeMonitor(self.input)
        # self.mon_I = spaic.StateMonitor(self.layer1, 'O')
        self.spk_l1 = spaic.SpikeMonitor(self.layer1)



        # self.monitor3 = spaic.StateMonitor(self.layer1, 'V')
        # self.monitor5 = spaic.StateMonitor(self.layer1, 'V')
        # self.monitor6 = spaic.StateMonitor(self.layer2, 'S')
        # 为输出层神经元组添加脉冲监视器。
        #self.monitor2 = spaic.SpikeMonitor(self.layer2)
        # self.monitor4 = spaic.SpikeMonitor(self.layer1)
        # 建立输出节点，并选择输出解码形式
        self.output = spaic.Decoder(num=10, coding_method='spike_counts',
            dec_target = self.layer1)
        
        # self.monitor5 = spaic.StateMonitor(self.layer2, 'O')

        # 加入学习算法，并选择需要训练的网络结构，（self代表全体ExampleNet结构）
        self.learner = spaic.Learner(algorithm = 'STCA', trainable=[self])
        self.learner.set_optimizer('Adam', 0.001)


# 创建训练数据集
from dataset.dataset import Dataset
# 训练参数
bat_size = 100
epoch_num = 1
learning_rate = 0.01
decay_step = 5
device = 'cpu'
# 训练集
train_set = Dataset(path=os.path.join(
                Path(__file__).resolve().parent.parent.parent, 'dataset'), is_train=True)
train_loader = spaic.Dataloader(train_set,
                                batch_size=bat_size,
                                shuffle=True,
                                drop_last=False)

test_set = Dataset(path=os.path.join(
                Path(__file__).resolve().parent.parent.parent, 'dataset'), is_train=False)
test_loader = spaic.Dataloader(test_set,
                                batch_size=bat_size,
                                shuffle=True,
                                drop_last=False)

# 后端编译
backend = spaic.Torch_Backend(device)
net = SNNModel(time=brain_component.get_run_time())
net.set_backend(backend)
net.set_backend_dt(dt=brain_component.get_dt())
net.build(backend)

param = net.get_testparams()
# 创建优化器对象，并传入网络模型的参数
optim = torch.optim.Adam(param, lr=learning_rate)  
scheduler = torch.optim.lr_scheduler.StepLR(optim, decay_step)

print("Start running", flush=True)
train_losses = []
train_acces = []
def train_epoch(epoch):
    epoch_loss = 0
    epoch_acc = 0
    for i, item in enumerate(train_loader):
        # 前向传播
        data, label = item
        data = brain_component.get_preprocessing('input')(data)
        net.input(data)
        net.run(brain_component.get_run_time())
        output = net.output.predict
        output = (output - torch.mean(output).detach()) / (
            torch.std(output).detach() + 0.1)
        label =  torch.tensor(label, device=device, dtype=torch.long)
        batch_loss = F.cross_entropy(output, label)
        # 反向传播
        batch_loss = batch_loss.requires_grad_()
        optim.zero_grad()
        batch_loss.backward(retain_graph=False)
        optim.step()
        # 记录误差
        epoch_loss += batch_loss.item()
        predict_labels = torch.argmax(output, 1)
        # 记录标签正确的个数
        num_correct = (predict_labels == label).sum().item()  
        batch_acc = num_correct / data.shape[0]
        epoch_acc += batch_acc
        print("Epoch: [{}] Batch: [{}/{}]  "\
            "Batch Acc: {:.4f}  "\
            "Batch Loss: {:.4f}  "\
            .format(epoch,i,len(train_loader),\
            batch_acc,batch_loss.item()), flush = True )
        break
    epoch_acc = epoch_acc / len(train_loader)
    epoch_loss = epoch_loss / len(train_loader)
    return epoch_acc, epoch_loss

for epoch in range(epoch_num):
    # 训练阶段
    print("Start training epoch[{}]".format(epoch), flush=True)
    train_acc, train_loss = train_epoch(epoch)
    train_acces.append(train_acc)
    train_losses.append(train_loss)

print("all_finish")

# 设置SNN模型
brain_component.set_SNN(net)
# 将训练后的网络保存到类脑构件配置中，量化前的网络保存在配置名字为0的目录下
bc_config_path = PathDPKRoot().brain_component(bc_name).config('0') # 获取config
print(bc_config_path)
brain_component.save(path=bc_config_path, save_weight=True)

######################################################
#量化流程
#以下是另一个阶段，通过load接口拿到训练后的网络
######################################################
from wuyuan import PathDPKRoot
from wuyuan import weight_quantify
from component.mnist_brain_component import MNISTBrainComponent
# load回网络,使用类脑构件的名字取回训练后保存的网络
bc_config_0 = PathDPKRoot().brain_component(bc_name).config('0')
bc_train = MNISTBrainComponent(bc_name)
bc_train.load(
    path=bc_config_0,
    device=device,
    load_weight=True)
# 量化

def print_net_param(net:spaic.Network, bit_width = 8):
    print("Starting to weight_quantify")
    for conn in net.get_connections():
        v_th = conn.post.parameters.get('v_th', 1.0)
        print("print_net_param:",v_th)

# print_net_param(bc_train.get_SNN())
# net_reload = weight_quantify(bc_train.get_SNN())
# print_net_param(bc_train.get_SNN())
net_reload = bc_train.get_SNN()
# 编译，生成二进制
from wuyuan import Compiler
bc_train.remove_SNN_implementation('darwin3')
compiler = Compiler(
    model = bc_train.get_SNN(),
    output_path = bc_train.add_SNN_implementation('darwin3'))
compiler.compile()
# 由于bins是存储在了临时目录，save接口保存到config下面
bc_config_1 = PathDPKRoot().brain_component(bc_name).config('1')
bc_train.save(
    path = bc_config_1,
    save_weight = True)
print("train finish.")
bc_train.load(
    path=bc_config_1,
    device=device,
    load_weight=True)