import time
from torchviz import make_dot
import random
import numpy as np
# import matplotlib.pyplot as plt

import argparse
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data.data_yanlong import train_dataset, test_dataset, data_update
from models import MLP_3, MLP_6
from lib.trans_all import *
from lib import IK, IK_loss, planner_loss
import torch
import torch.nn as nn
import math
import os
from lib.save import checkpoints
from lib.plot import *


class main():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Training MLP")
        self.parser.add_argument('--batch_size', type=int, default=5, help='input batch size for training (default: 1)')
        self.parser.add_argument('--learning_rate', type=float, default=0.0025, help='learning rate (default: 0.003)')
        self.parser.add_argument('--epochs', type=int, default=100, help='gradient clip value (default: 300)')
        self.parser.add_argument('--clip', type=float, default=1, help='gradient clip value (default: 1)')
        self.parser.add_argument('--num_train', type=int, default=1000)
        self.args = self.parser.parse_args()

        # 使用cuda!!!!!!!!!!!!!!!未补齐
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 训练集数据导入
        # self.data_train = TensorDataset(data_update.train_data(self.args.num_train))
        self.data_train = TensorDataset(train_dataset.a[:self.args.num_train])
        self.data_loader_train = DataLoader(self.data_train, batch_size=self.args.batch_size, shuffle=False)
        # 测试集数据导入
        self.data_test = TensorDataset(test_dataset.c, test_dataset.c)
        self.data_loader_test = DataLoader(self.data_test, batch_size=self.args.batch_size, shuffle=False)

        # 定义训练权重保存文件路径
        self.checkpoint_dir = r'/home/cn/RPSN_2/work_dir/test16'
        # 多少伦保存一次
        self.num_epoch_save = 200

        # 选择模型及参数
        self.num_i = 12
        self.num_h = 100
        self.num_o = 6
        self.model = MLP_3
        
        # 如果是接着训练则输入前面的权重路径
        self.model_path = r''

        # 定义DH参数
        self.link_length = torch.tensor([0, -0.6127, -0.57155, 0, 0, 0])  # link length
        self.link_offset = torch.tensor([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])  # link offset
        self.link_twist = torch.FloatTensor([math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0])

        # 定义放置位置
        self.ori_position = torch.FloatTensor([-0.2601253604477298, -1.3122437566888538, -2.264208369694192, -3.4790699458823013, 2.8052134780208995, -0.8103880217657344])
    def train(self):
        num_i = self.num_i
        num_h = self.num_h
        num_o = self.num_o

        NUMError1 = []
        NUMError2 = []
        NUM_incorrect = []
        NUM_correct = []
        NUM_correct_test = []
        NUM_incorrect_test = []
        echo_loss = []
        echo_loss_test = []
        NUM_2_to_1 = []
        NUM_mid = []
        NUM_lar = []
        NUM_sametime_solution = []

        epochs = self.args.epochs
        data_loader_train = self.data_loader_train
        learning_rate = self.args.learning_rate
        model = self.model.MLP_self(num_i , num_h, num_o) 
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=0.000)  # 定义优化器
        model_path = self.model_path

        if os.path.exists(model_path):          
            checkpoint = torch.load(model_path)  
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            loss = checkpoint['loss']
            print(f"The loading model is complete, let's start this training from the {start_epoch} epoch, the current loss is : {loss}")
        else:
            print("There is no pre-trained model under the path, and the following training starts from [[epoch1]] after random initialization")
            start_epoch = 1

        # 开始训练
        for epoch in range(start_epoch , start_epoch + epochs):
  
            sum_loss = 0.0
            sum_loss_test = 0.0
            numError1 = 0
            numError2 = 0
            num_incorrect = 0
            num_correct = 0

            for data in data_loader_train:  # 读入数据开始训练
                # 将目标物体1x6与放置位置1x6组合为1x12
                inputs = shaping_inputs_6to12(self.ori_position, data[0])

                intermediate_outputs = model(inputs)

                # 将1x12输入转为10x1x6,
                input_tar = shaping_inputs_12to6(inputs) # 得到变换矩阵
                # 得到每个1x6的旋转矩阵
                input_tar = shaping(input_tar)
                # 将网络输出1x6转换为1x3
                intermediate_outputs = shaping_output_6to3(intermediate_outputs)

                outputs = torch.empty((0, 6)) # 创建空张量
                for each_result in intermediate_outputs: # 取出每个batch_size中的每个数据经过网络后的结果1x3
                    pinjie1 = torch.cat([each_result, torch.zeros(1).detach()])
                    pinjie2 = torch.cat([torch.zeros(2).detach(), pinjie1])
                    outputs = torch.cat([outputs, pinjie2.unsqueeze(0)], dim=0)

                intermediate_outputs.retain_grad()
                outputs.retain_grad()

                MLP_output_base = shaping(outputs)  # 对输出做shaping运算-1X6变为4X4
                MLP_output_base.retain_grad()

                # 计算 IK_loss_batch
                IK_loss_batch = torch.tensor(0.0, requires_grad=True)
                IK_loss2 = torch.tensor(0.0, requires_grad=True)
                IK_loss3 = torch.tensor(0.0, requires_grad=True)

                for i in range(len(input_tar)):

                    angle_solution, num_Error1, num_Error2, the_NANLOSS_of_illegal_solution_with_num_and_Nan = IK.calculate_IK(
                        input_tar[i], 
                        MLP_output_base[i], 
                        self.link_length, 
                        self.link_offset, 
                        self.link_twist)

                    if not num_Error1 + num_Error2 == 0:
                        IK_y_o_n_tar = 0
                    else:
                        IK_y_o_n_tar = 1

                    # 存在错误打印
                    numError1 = numError1 + num_Error1
                    numError2 = numError2 + num_Error2
                    # 计算单IK_loss
                    IK_loss1, num_NOError1, num_NOError2 = IK_loss.calculate_IK_loss(angle_solution, the_NANLOSS_of_illegal_solution_with_num_and_Nan)

                    #计算plannerloss/目标物体位置和放置位置同时有解
                    llll = int(len(input_tar) / 2)
                    if i in range(llll):
                        angle_solution_ori, IK_y_or_n_ori = planner_loss.IK_yes_or_no(
                            input_tar[i + llll], 
                            MLP_output_base[i + llll], 
                            self.link_length, 
                            self.link_offset, 
                            self.link_twist)
                        if not IK_y_o_n_tar + IK_y_or_n_ori == 2:
                            IK_loss2 = IK_loss2 + 100

                    # 总loss
                    IK_loss_batch = IK_loss_batch + IK_loss1 + IK_loss2
                    # IK_loss_batch = IK_loss_batch + IK_loss1

                    # 右/无错误打印
                    num_incorrect = num_incorrect + num_NOError1
                    num_correct = num_correct + num_NOError2
                # print(IK_loss_batch)                            
                # planner_loss//相距较远的不进行loss计算，较近的添加loss，并且如果求出的两个底盘位置距离在一定范围则转换为一个中间位置
                llll = int(len(input_tar) / 2)
                for p in range(llll):
                    obj_base = outputs[p]
                    tar_base = outputs[p + llll]
                    x_1 = tar_base[3]
                    y_1 = tar_base[4]
                    x_2 = obj_base[3]
                    y_2 = obj_base[4]
                    distance1 = math.sqrt((y_2 - y_1)**2 + (x_2 - x_1)**2)
                    obj_input = inputs[p]
                    x_1_in = obj_input[3]
                    y_1_in = obj_input[4]
                    x_2_in = obj_input[9]
                    y_2_in = obj_input[10]
                    distance2 = math.sqrt((y_2_in - y_1_in)**2 + (x_2_in - x_1_in)**2)
                    # print(distance1, distance2)
                    if distance2 > 2.6:
                        if distance1 > distance2:
                            IK_loss3 = IK_loss3 + (distance1 - distance2) * 100
                        else:
                            IK_loss3 = IK_loss3 + torch.tensor([0])
                    else:
                        # if distance1 >= 3:
                        #     IK_loss3 = IK_loss3 + torch.tensor([0]) 
                        # elif 1.5 < distance1 < 2.5:
                        #     IK_loss3 = IK_loss3 + distance1 * cos(torch.tensor(math.pi/2 * (distance1-3)/1.5))
                        # else:
                        #     IK_loss3 = IK_loss3 + torch.tensor([0]) - 1.5 * sin(torch.tensor(math.pi / 2 * (-(2/3 * distance1) + 1)))
                        # if not distance1 > 0.1:
                        #     IK_loss3 = IK_loss3 + distance1 * 10
                        # else:
                        #     IK_loss3 = IK_loss3 + (10 * distance1 - 1)
                # print(IK_loss3, '*' * 50)
                        IK_loss3 = IK_loss3 + (100 * distance1 - 10) * 100
                # print(IK_loss3, '*' * 50)
                IK_loss_batch = IK_loss_batch + IK_loss3
                # print(IK_loss_batch, '*' * 50)

                IK_loss_batch.retain_grad()

                optimizer.zero_grad()  # 梯度初始化为零，把loss关于weight的导数变成0

                # 定义总loss函数
                loss = (IK_loss_batch) / len(inputs)
                loss.retain_grad()

                # 记录x轮以后网络模型checkpoint，用来查看数据流
                if epoch % self.num_epoch_save == 0:
                    # print("第{}轮的网络模型被成功存下来了！储存内容包括网络状态、优化器状态、当前loss等".format(epoch))
                    checkpoints(model, epoch, optimizer, loss, self.checkpoint_dir)

                loss.backward()  # 反向传播求梯度
                # loss.backward(torch.ones_like(loss))  # 反向传播求梯度
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.args.clip)  # 进行梯度裁剪
                optimizer.step()  # 更新所有梯度
                sum_loss = sum_loss + loss.data

            echo_loss.append(sum_loss / (len(data_loader_train)))
            # print(echo_loss)
            
            NUMError1.append(numError1)
            NUMError2.append(numError2)
            NUM_incorrect.append(num_incorrect)
            NUM_correct.append(num_correct)

            print("numError1", numError1)
            print("numError2", numError2)
            print("num_correct", num_correct)
            print("num_incorrect", num_incorrect)


            model.eval()

            data_loader_test = self.data_loader_test
            num_incorrect_test = 0
            num_correct_test = 0
            num_2_to_1 = 0
            num_mid = 0
            num_lar = 0
            num_sametime_solution = 0
            num_distance_large = 0
            for data_test in data_loader_test:
                with torch.no_grad():
                    inputs_test = shaping_inputs_6to12(self.ori_position, data_test[0])
                    intermediate_outputs_test = model(inputs_test)
                    input_tar_test = shaping_inputs_12to6(inputs_test)
                    input_tar_test = shaping(input_tar_test)
                    intermediate_outputs_test = shaping_output_6to3(intermediate_outputs_test)
                    outputs_test = torch.empty((0, 6))
                    for each_result in intermediate_outputs_test:
                        pinjie1 = torch.cat([each_result, torch.zeros(1).detach()])
                        pinjie2 = torch.cat([torch.zeros(2).detach(), pinjie1])
                        outputs_test = torch.cat([outputs_test, pinjie2.unsqueeze(0)], dim=0)

                    MLP_output_base_test = shaping(outputs_test)

                    # 计算 IK_loss_batch
                    IK_loss_batch_test = torch.tensor(0.0, requires_grad=True)
                    IK_loss3_test = torch.tensor(0.0, requires_grad=True)
                    for i in range(len(input_tar_test)):
                        angle_solution = IK.calculate_IK_test(
                            input_tar_test[i], 
                            MLP_output_base_test[i], 
                            self.link_length, 
                            self.link_offset, 
                            self.link_twist)
                        # IK时存在的错误打印
                        IK_loss_test1, IK_loss_test_incorrect, IK_loss_test_correct = IK_loss.calculate_IK_loss_test(angle_solution)
                        # 计算IK_loss时存在的错误与正确的打印
                        num_incorrect_test = num_incorrect_test + IK_loss_test_incorrect
                        num_correct_test = num_correct_test + IK_loss_test_correct
                        # 计算IK_loss
                        IK_loss_batch_test = IK_loss_batch_test + IK_loss_test1

                    # 计算打印信息
                    llll = int(len(input_tar) / 2)
                    for i in range(llll):
                        obj_base = outputs_test[i]
                        tar_base = outputs_test[i + llll]
                        x_1 = tar_base[3]
                        y_1 = tar_base[4]
                        x_2 = obj_base[3]
                        y_2 = obj_base[4]
                        distance = math.sqrt((y_2 - y_1)**2 + (x_2 - x_1)**2)
                        obj_input = inputs_test[i]
                        x_1_in = obj_input[3]
                        y_1_in = obj_input[4]
                        x_2_in = obj_input[9]
                        y_2_in = obj_input[10]
                        distance2 = math.sqrt((y_2_in - y_1_in)**2 + (x_2_in - x_1_in)**2)
                        # print(distance1, distance2)
                        # if not distance2 > 2.6:
                        #     num_distance_large += 1
                        if distance >= 1:
                            # IK_loss3_test = IK_loss3_test + torch.tensor([0])
                            num_lar = num_lar + 1
                        elif 0.5 < distance < 1:
                            # IK_loss3_test = IK_loss3_test + distance * sin(torch.tensor(math.pi / 2 * (distance-1.5)/4.5)) * 10
                            num_mid = num_mid + 1
                        else:
                            # IK_loss3_test = IK_loss3_test + torch.tensor([0])
                            num_2_to_1 = num_2_to_1 + 1

                                        # #计算plannerloss/目标物体位置和放置位置同时有解
                    
                    for aaaa in range(llll):
                        angle_solution_ori, IK_y_or_n_ori = planner_loss.IK_yes_or_no(
                            input_tar_test[aaaa], 
                            MLP_output_base_test[aaaa], 
                            self.link_length, 
                            self.link_offset, 
                            self.link_twist)
                        angle_solution_tar, IK_y_or_n_tar = planner_loss.IK_yes_or_no(
                            input_tar_test[aaaa + llll], 
                            MLP_output_base_test[aaaa + llll], 
                            self.link_length, 
                            self.link_offset, 
                            self.link_twist)
                        if IK_y_or_n_tar + IK_y_or_n_ori == 2:
                            num_sametime_solution += 1

            #         IK_loss_batch_test = IK_loss_batch_test + IK_loss3_test

            #         # 定义总loss函数
            #         loss = (IK_loss_batch_test) / len(inputs_test)

            #         sum_loss_test = sum_loss_test + loss.data

            # echo_loss_test.append(sum_loss_test / len(data_loader_test))
            # print(num_distance_large)
            print("num_correct_test", num_correct_test)
            print("num_incorrect_test", num_incorrect_test)
            print('num_2_to1', num_2_to_1)
            print('num_sametime_solution', num_sametime_solution)

            NUM_2_to_1.append(num_2_to_1)
            NUM_mid.append(num_mid)
            NUM_lar.append(num_lar)
            NUM_incorrect_test.append(num_incorrect_test)
            NUM_correct_test.append(num_correct_test)
            NUM_sametime_solution.append(num_sametime_solution)

            print('[%d,%d] loss:%.03f' % (epoch, start_epoch + epochs-1, sum_loss / (len(data_loader_train))), "-" * 100)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        # 画图
        plot_IK_solution(self.checkpoint_dir, start_epoch, epochs, len(self.data_test), NUM_incorrect_test, NUM_correct_test)
        plot_train(self.checkpoint_dir, start_epoch, epochs, self.args.num_train, NUMError1, NUMError2, NUM_incorrect, NUM_correct)
        plot_train_loss(self.checkpoint_dir, start_epoch, epochs, echo_loss)
        # plot_test_loss(self.checkpoint_dir, start_epoch, epochs, echo_loss_test)
        plot_2_to_1(self.checkpoint_dir, start_epoch, epochs, NUM_2_to_1, NUM_mid, NUM_lar)
        plot_sametime_solution(self.checkpoint_dir, start_epoch, epochs, NUM_sametime_solution)

if __name__ == "__main__":
    a = main()
    a.train()
