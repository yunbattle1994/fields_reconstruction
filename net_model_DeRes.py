import time
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt


from Utilize import data_visualization as visual, results_evaluation as result, thermal_evaluation_gpu as thermal


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            # m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            # m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        # elif isinstance(m, nn.BatchNorm2d):
        #     m.weight.data.normal_(0)
        #     m.bias.data.zero_()



class Weighted_field_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,weight, x, y):

        L2 = ((x-y).pow(2)).sqrt()


        loss = torch.mean(L2 * weight)

        return loss

class Gradient_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        grad_x1 = Tensor.abs(x[:, :, 1:, :] - x[:, :, 0:-1, :])
        grad_x2 = Tensor.abs(x[:, :, :, 1:] - x[:, :, :, 0:-1])

        grad_y1 = Tensor.abs(y[:, :, 1:, :] - y[:, :, 0:-1, :])
        grad_y2 = Tensor.abs(y[:, :, :, 1:] - y[:, :, :, 0:-1])

        loss = Tensor.sum(Tensor.abs(grad_x1 - grad_y1)) + Tensor.sum(Tensor.abs(grad_x2 - grad_y2))

        return (loss / (Tensor.numel(grad_x1) + Tensor.numel(grad_x2)))


class Gradient_loss_2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        grad_x1 = Tensor.abs(x[:, :, 2:, :] - x[:, :, 0:-2, :]) / 2
        grad_x2 = Tensor.abs(x[:, :, :, 2:] - x[:, :, :, 0:-2]) / 2

        grad_y1 = Tensor.abs(y[:, :, 2:, :] - y[:, :, 0:-2, :]) / 2
        grad_y2 = Tensor.abs(y[:, :, :, 2:] - y[:, :, :, 0:-2]) / 2

        loss = Tensor.sum(Tensor.abs(grad_x1 - grad_y1)) + Tensor.sum(Tensor.abs(grad_x2 - grad_y2))

        return (loss / (Tensor.numel(grad_x1) + Tensor.numel(grad_x2)))


class model(object):

    def __init__(self, model_name, input_size, hidden_size=256, output_size=7, mode='DG'):

        """ net_name: 神经网络名称
            input_size: 神经网络的输入尺寸：channel, height, width
            hidden_size: 是否使用多GPU并行
        """

        self.model_name = model_name
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        # 根据所需要网络自行添加

        if model_name == 'DeRes18' or "DeRes":
            import DeResNet
            import CoResNet
            self.G_net = DeResNet.DeResNet18(nc=input_size[1],  ngf=16, nz=output_size + hidden_size)
            self.D_net = CoResNet.CoResNet18(nz=hidden_size, nc=2, mode=mode, is_class=False)
        elif model_name == 'DeRes34':
            import DeResNet
            import CoResNet
            self.G_net = DeResNet.DeResNet34(nc=input_size[1],  ngf=16, nz=output_size + hidden_size)
            self.D_net = CoResNet.CoResNet34(nz=hidden_size, nc=2, mode=mode, is_class=False)

        self.G_net.apply(initialize_weights)
        self.D_net.apply(initialize_weights)

        # 记录过程变量
        self.process_info = {"train_time": [], "eval_time": [], "iteration": [], "learning_rate": [],
                             "train_loss_logs": [], "eval_loss_logs": [],
                             "train_field_errors": [], "eval_field_errors": [],
                             "train_result_scores": [], "eval_result_scores": [],
                             "train_flow_results": [], "eval_flow_results": [],
                             "lowest_field_loss": [], "best_Nu_scores": [],
                             "post_time": [], "loss_weight": [],
                             }


    def net_set(self, save_name,  model_name="DeRes", field_name=("T",), field_index=(1,), field_bound=(0, 1),
                loss_weight=(0, 1, 0),
                device_name=torch.device('cuda:' + str(0)), is_parallel=False, is_resume=False,):

        # 计算设备
        self.device = device_name

        if is_parallel:
            # 设置gpu并行
            self.G_net = nn.DataParallel(self.G_net).to(self.device)
            self.D_net = nn.DataParallel(self.D_net).to(self.device)

        else:
            self.G_net = self.G_net.to(self.device)
            self.D_net = self.D_net.to(self.device)

        # 交叉熵损失
        self.F_loss = nn.MSELoss().to(self.device)
        # self.F_loss = Weighted_field_loss().to(self.device)
        self.G_loss = Gradient_loss().to(self.device)

        # 创建模型保存路径
        self.save_path = save_name + "//" + model_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.model_path = self.save_path + "//model"
        self.predict_path = self.save_path + "//pred"
        self.log_path = self.save_path + "//log"
        if not os.path.exists(self.predict_path):
            os.makedirs(self.predict_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # 场名称
        self.field_name = field_name
        # 场索引
        self.field_index = field_index
        # 场的归一化数据
        self.field_bound = field_bound
        # 损失函数权重
        self.loss_weight = loss_weight


        # 记录文件夹
        self.writer = visual.tensorboard_vision(self.log_path)
        self.logger = visual.matplotlib_vision(self.log_path)

        # 记录图结构
        x = torch.randn(10, 2, self.input_size[2], self.input_size[3]).to(self.device)
        y = torch.randn(10, self.output_size + self.hidden_size).to(self.device)
        self.writer.graph(self.D_net, x)
        self.writer.graph(self.G_net, y)

        # 存储模型载入
        if is_resume == True:
            self.is_load = self.load_model()
        else:
            self.is_load = False

    def train(self, train_loader, eval_loader, test_loader=None,
              learning_rate=0.1, boundary_epochs=(60, 100, 140),
              rates_decay=(1, 0.1, 0.01, 0.001), weight_decay=0, grad_rates=(1, 0.1, 0.01),
              start_epoch=0, end_epoch=200, display_batch=100,
              *args, **kwargs):


        " 训练参数初始化 "
        if self.is_load == True:
            start_epoch = len(self.process_info['iteration'])
            self.best_acc = self.process_info['best_Nu_scores'][-1]
            self.low_loss = self.process_info['lowest_field_loss'][-1]
            self.process_info["iteration"] = list(range(start_epoch))
        else:
            self.best_acc = -100.
            self.low_loss = 100.

        # 学习率设置函数
        self.lr_fn = self.set_learning_rate(initial_learning_rate=learning_rate,
                                            boundary_epochs=boundary_epochs, rates_decay=rates_decay)
        # grad_loss_weight设置函数
        self.grad_weight = self.set_learning_rate(initial_learning_rate=self.loss_weight[-1],
                                            boundary_epochs=boundary_epochs, rates_decay=grad_rates)

        # 优化器设置
        self.G_optimizer = torch.optim.Adam(self.G_net.parameters(), lr=0.001, weight_decay=weight_decay)
        self.D_optimizer = torch.optim.Adam(self.D_net.parameters(), lr=0.001, weight_decay=weight_decay, )

        # self.G_optimizer = torch.optim.SGD(self.G_net.parameters(), lr=0.001, momentum=0.9)
        # self.D_optimizer = torch.optim.SGD(self.G_net.parameters(), lr=0.001, momentum=0.9)

        " 开始训练循环 "
        for epoch in range(start_epoch, end_epoch):
            # 记录epoch开始的时间
            epoch_start_time = time.time()

            print('start epoch [{:04d} / {:04d}], learning_rate {:.5f} '.format(epoch, end_epoch, self.lr_fn(epoch)))

            # 训练学习率设置
            for param_group in self.G_optimizer.param_groups:
                param_group['lr'] = self.lr_fn(epoch)
            for param_group in self.D_optimizer.param_groups:
                param_group['lr'] = self.lr_fn(epoch)

            end_batch = 128*2 #len(train_loader)

            self.loss_weight[1] = self.grad_weight(epoch)

            for batch in range(end_batch):
                # 训练模式切换
                self.D_net.train()
                self.G_net.train()

                # 数据载入
                name, input, grid, field, heflx, design, flowres_true = next(iter(train_loader))
                batch_size = name.shape[0]
                field_real = field.to(self.device)

                grid = grid.to(self.device)
                # grid 简化
                redu_grid = self.D_net(grid)
                input = input.to(self.device)
                input_real = torch.cat([input, redu_grid], dim=1)
                # input_real = input_real.to(self.device)
                # 模型预测
                field_pred = self.G_net(input_real)

                loss_1 = self.F_loss(field_pred, field_real)
                loss_2 = self.G_loss(field_pred, field_real)

                loss_batch = self.loss_weight[0] * loss_1 + self.loss_weight[1] * loss_2
                # 梯度更新
                self.D_optimizer.zero_grad()
                self.G_optimizer.zero_grad()
                # 反向传播
                loss_batch.backward()
                # 梯度下降
                self.G_optimizer.step()
                self.D_optimizer.step()
                # batch时间记录
                batch_end_time = time.time()

                # 进行评估和输出
                if (batch > 0 and batch % display_batch == 0) or (batch == end_batch - 1):
                    # 模型验证集评估
                    eval_start_time = time.time()

                    train_loss_logs, train_errors, train_flow_results,_ = self.evaluation(train_loader)
                    eval_loss_logs, eval_errors, eval_flow_results,_ = self.evaluation(eval_loader)
                    # loss 统计
                    (train_Total_loss, train_Field_loss, train_Grad_loss) = train_loss_logs
                    (eval_Total_loss, eval_Field_loss, eval_Grad_loss) = eval_loss_logs
                    # field_error 统计
                    train_field_errors = np.mean(np.array(train_errors), axis=0)
                    eval_field_errors = np.mean(np.array(eval_errors), axis=0)
                    # Nu, f, Tb 统计
                    train_result_scores = train_flow_results.r2_score()[:, np.newaxis].max(axis=1, initial=-1)
                    eval_result_scores = eval_flow_results.r2_score()[:, np.newaxis].max(axis=1, initial=-1)
                    # eval时间记录
                    eval_end_time = time.time()
                    # epoch时间记录
                    epoch_end_time = time.time()
                    # 训练效果输出
                    print(
                        "Batch [{:04d} / {:04d}] | Eval Time {:.4f} | Epoch Time {:.4f} | "
                        "Lowest field Loss {:.2e} | Best Nu scores {:.2e} \n"
                        "Train total Loss {:.2e} | Eval total Loss {:.2e} | "
                        "Train FIELD Loss {:.2e} | Eval FIELD Loss {:.2e} | "
                        "Train GRAD Loss {:.2e} | Eval GRAD Loss {:.2e}  | \n"
                        "Train field max errors {:s} | Eval field max errors {:s} | \n"
                        "Train flow scores {:s} | Eval flow scores {:s} ".format(
                            batch, end_batch, epoch_end_time - eval_start_time, epoch_end_time - epoch_start_time,
                            self.low_loss, self.best_acc,
                            train_Total_loss, eval_Total_loss,
                            train_Field_loss, eval_Field_loss,
                            train_Grad_loss, eval_Grad_loss,
                            str(train_field_errors[:, 2]), str(eval_field_errors[:, 2]),
                            str(train_result_scores), str(eval_result_scores),
                        )
                    )

            " 后处理计算 "
            post_start_time = time.time()

            " 训练过程记录 "
            # 统计学习率
            self.process_info["iteration"].append(epoch)
            self.process_info["learning_rate"].append(self.lr_fn(epoch))
            self.process_info["train_time"].append(epoch_end_time - epoch_start_time)
            self.process_info["eval_time"].append(eval_end_time - eval_start_time)
            # 统计最佳损失及最高精度
            self.process_info["lowest_field_loss"].append(self.low_loss)
            self.process_info["best_Nu_scores"].append(self.best_acc)
            # 统计混合损失及模型得分
            self.process_info["train_loss_logs"].append(train_loss_logs)
            self.process_info["eval_loss_logs"].append(eval_loss_logs)
            self.process_info["train_field_errors"].append(train_field_errors)
            self.process_info["eval_field_errors"].append(eval_field_errors)
            self.process_info['train_flow_results'].append([train_flow_results.true, train_flow_results.pred])
            self.process_info['eval_flow_results'].append([eval_flow_results.true, eval_flow_results.pred])
            self.process_info["train_result_scores"].append(train_result_scores)
            self.process_info["eval_result_scores"].append(eval_result_scores)

            "训练模型保存"
            self.save_model(epoch, record_pattern=20)

            " 训练过程可视化tensorboard "
            self.tensorboard_visual(epoch)

            if epoch % 20 == 0:

                " 训练过程可视化matplotlib "
                self.matplotlib_visual(epoch)

                " 预测test结果，输出tecplot"
                self.predict(test_loader, epoch=epoch)

            "在分界点重新载入模型"
            self.load_model(resume_pattern="lowest", epoch=epoch, boundary_epochs=boundary_epochs)

            post_end_time = time.time()
            print('end epoch [{:04d} / {:04d}] | Post Time {:.4f} \n'.format(epoch, end_epoch,
                                                                             post_end_time - post_start_time))



    def evaluation(self, dataset_loader, mode="train"):

        self.G_net.eval()
        self.D_net.eval()

        Total_loss = result.LogMeter()
        Field_loss = result.LogMeter()
        Grad_loss = result.LogMeter()

        flow_results = result.AccMeter()
        name_log = result.AccMeter()
        field_error = []

        if mode == "train":
            len_data = len(dataset_loader)
        else:
            len_data = len(dataset_loader)

        with torch.no_grad():
            for _ in range(len_data):
                # 数据载入
                name, input, grid, field, heflx, design, flowres_true = next(iter(dataset_loader))
                batch_size = name.shape[0]
                field_real = field.to(self.device)
                grid = grid.to(self.device)
                # grid 简化
                redu_grid = self.D_net(grid)
                input = input.to(self.device)
                input_real = torch.cat([input, redu_grid], dim=1)
                # input_real = input_real.to(self.device)
                # 模型预测
                field_pred = self.G_net(input_real)
                loss_1 = self.F_loss(field_pred, field_real)
                loss_2 = self.G_loss(field_pred, field_real)
                # loss_batch = loss_1 + loss_2

                loss_batch = self.loss_weight[0] * loss_1 + self.loss_weight[1] * loss_2
                # 统计场的精度
                L_error = result.field_error(field_real, field_pred)
                # 统计准则数精度
                flowres_pred = thermal.cal_all(self.field_name, self.field_bound, field_pred, grid.to(self.device),
                                               heflx.to(self.device), design.to(self.device), device=self.device)
                flowres_true = thermal.cal_all(self.field_name, self.field_bound, field_real, grid.to(self.device),
                                               heflx.to(self.device), design.to(self.device), device=self.device)
                field_error.extend(L_error)

                flow_results.update(y_true=np.array(flowres_true), y_pred=np.array(flowres_pred))
                name_log.update(y_true=np.array(name), y_pred=np.array(name))

                Total_loss.update(float(loss_batch.item()))
                Field_loss.update(float(loss_1.item()))
                Grad_loss.update(float(loss_2.item()))

        loss_log = [Total_loss.avg, Field_loss.avg, Grad_loss.avg]

        return loss_log, field_error, flow_results, name_log

    def predict(self, test_loader, epoch=0):
        # self.encoder_net.eval()
        self.G_net.eval()
        self.D_net.eval()

        with torch.no_grad():

            for i, (name, input, grid, field, heflx, design, flowres_true) in enumerate(test_loader):
                batch_size = name.shape[0]

                field_real = field.to(self.device)
                grid_real = grid.to(self.device)
                input = input.to(self.device)
                # grid 简化
                redu_grid = self.D_net(grid_real)
                input_real = torch.cat([input, redu_grid], dim=1)
                # input_real = input_real.to(self.device)
                # 模型预测
                field_pred = self.G_net(input_real).detach().cpu()

                self.output_field(name.numpy(), grid.numpy(), field.numpy(), field_pred.numpy(), epoch)

            return

    def feature_extract(self, dataset_loader):


        self.G_net.eval()

        with torch.no_grad():
            for i, (visual_inputs, visual_names) in enumerate(dataset_loader):
            # for batch_index in range(10):    #batch_size=32
            #     visual_inputs, visual_labels = next(iter(dataset_loader))   #next()获得iterator对象
                if i != len(dataset_loader)-1:
                    pass
                else:
                    visual_inputs_ = visual_inputs.cuda()
                    visual_features = self.G_net.extract(visual_inputs_)    #函数初始化
                    for index, feature in enumerate(visual_features):   #遍历各层特征
                        # feature = feature.cpu().detach().numpy()
                        # 初始化特征输出feature_output
                        name = visual_names[0]
                        feature = feature[0]
                        save_path = self.predict_path + "\\visual_names_" + str(name) + "\\layer" + str(index)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        feature = feature.cpu().detach().numpy()
                        self.logger.features_output(feature, visual_channel=12, save_path=save_path)




    def save_model(self, epoch, record_pattern=None):

        # 最新模型
        torch.save({
            'D_net_state_dict': self.D_net.state_dict(),
            'G_net_state_dict': self.G_net.state_dict(),
            'process_info': self.process_info},
            os.path.join(self.model_path, 'latest_model.pth'))

        " 训练最佳模型保存 "
        # 最低平均误差模型
        if self.process_info["eval_loss_logs"][epoch][1] < self.low_loss:
            torch.save({'D_net_state_dict': self.D_net.state_dict(),
                        'G_net_state_dict': self.G_net.state_dict(),
                        'process_info': self.process_info},
                       os.path.join(self.model_path, 'lowest_loss_model.pth'))
            self.low_loss = self.process_info["eval_loss_logs"][epoch][1]
        # 最高精度模型
        if self.process_info["eval_result_scores"][epoch][0] > self.best_acc:
            torch.save({'D_net_state_dict': self.D_net.state_dict(),
                        'G_net_state_dict': self.G_net.state_dict(),
                        'process_info': self.process_info},
                       os.path.join(self.model_path, 'best_accs_model.pth'))
            self.best_acc = self.process_info["eval_result_scores"][epoch][0]

        torch.save({'D_net_state_dict': self.D_net.state_dict(),
                    'G_net_state_dict': self.G_net.state_dict(),
                    'process_info': self.process_info},
                   os.path.join(self.model_path, 'latest_model.pth'))

        if record_pattern != None:

            if epoch % record_pattern == 0:
                torch.save({'D_net_state_dict': self.D_net.state_dict(),
                            'G_net_state_dict': self.G_net.state_dict(),
                            'process_info': self.process_info},
                           os.path.join(self.model_path, 'epoch_' + str(epoch) + '_model.pth'))


    def load_model(self, resume_pattern=None, epoch=None, boundary_epochs=None, model_name=None):

        if resume_pattern == "direct":
            if os.path.isfile(self.model_path + '//' + model_name):
                print("=> loading breakpoint '{}'".format(model_name))
                breakpoint = torch.load(self.model_path + '//' + model_name)
                self.D_net.load_state_dict(breakpoint['D_net_state_dict'])
                self.G_net.load_state_dict(breakpoint['G_net_state_dict'])
                return True
            else:
                print("=> no breakpoint found, start a new model")
                return False
        elif resume_pattern == "lowest":
            # 判断是否到学习率下降的步数
            threshold = True if epoch in boundary_epochs else False
            if threshold:
                print("=> loading breakpoint '{}'".format("lowest_loss_model"))
                breakpoint = torch.load(self.model_path + "//lowest_loss_model.pth")
                self.D_net.load_state_dict(breakpoint['D_net_state_dict'])
                self.G_net.load_state_dict(breakpoint['G_net_state_dict'])
                return True
        elif resume_pattern == "best":
            # 判断是否到学习率下降的步数
            threshold = True if epoch in boundary_epochs else False
            if threshold:
                print("=> loading breakpoint '{}'".format("best_scores_model"))
                breakpoint = torch.load(self.model_path + "//best_scores_model.pth")
                self.D_net.load_state_dict(breakpoint['D_net_state_dict'])
                self.G_net.load_state_dict(breakpoint['G_net_state_dict'])
                return True
        elif resume_pattern == None:
            # 如果是None则为resume model
            if os.path.isfile(self.model_path + "//latest_model.pth"):
                print("=> loading breakpoint '{}'".format("latest_model"))
                breakpoint = torch.load(self.model_path + "//latest_model.pth")
                # self.D_net.load_state_dict(breakpoint['D_net_state_dict'])
                self.G_net.load_state_dict(breakpoint['G_net_state_dict'])
                # resume模型时需要载入process_info的参数
                self.process_info = breakpoint['process_info']
                print("=> loaded breakpoint (epoch {})".format(len(self.process_info['iteration'])))
                return True
            else:
                print("=> no breakpoint found, start a new model")
                return False

    def matplotlib_visual(self, epoch):

        plot_channel = 0
        # matplotlib绘制训练过程图
        plt.figure(plot_channel, figsize=(20, 10))
        plt.clf()
        # plt.ion()  # interactive mode on

        Loss_log_t = np.array(self.process_info["train_loss_logs"])
        Loss_log_e = np.array(self.process_info["eval_loss_logs"])

        plt.subplot(221)
        plt.xscale('linear')
        plt.yscale('log')
        self.logger.scalar(self.process_info["iteration"], Loss_log_t[:, 0], title="loss", label="train_total")
        self.logger.scalar(self.process_info["iteration"], Loss_log_e[:, 0], title="loss", label="eval_total")
        self.logger.scalar(self.process_info["iteration"], Loss_log_t[:, 1], title="loss", label="train_Field_loss")
        self.logger.scalar(self.process_info["iteration"], Loss_log_e[:, 1], title="loss", label="eval_Field_loss")
        self.logger.scalar(self.process_info["iteration"], Loss_log_t[:, 2], title="loss", label="train_Grad_loss")
        self.logger.scalar(self.process_info["iteration"], Loss_log_e[:, 2], title="loss", label="eval_Grad_loss")
        # ax.yaxis.grid(True, which='minor')

        plt.subplot(222)
        plt.xscale('linear')
        plt.yscale('log')
        self.logger.scalar(self.process_info["iteration"], Loss_log_t[:, 0], title="loss", label="train_Total_loss")
        self.logger.scalar(self.process_info["iteration"], Loss_log_e[:, 0], title="loss", label="eval_Total_loss")
        # ax.yaxis.grid(True, which='minor')

        plt.subplot(223)
        plt.xscale('linear')
        plt.yscale('log')
        self.logger.scalar(self.process_info["iteration"], Loss_log_t[:, 1] / self.loss_weight[0],
                           title="loss", label="train_Field_loss")
        self.logger.scalar(self.process_info["iteration"], Loss_log_e[:, 1] / self.loss_weight[0],
                           title="loss", label="train_Field_loss")

        plt.subplot(224)
        plt.xscale('linear')
        plt.yscale('log')
        self.logger.scalar(self.process_info["iteration"], Loss_log_t[:, 2] / self.loss_weight[1],
                           title="loss", label="train_Grad_loss")
        self.logger.scalar(self.process_info["iteration"], Loss_log_e[:, 2] / self.loss_weight[1],
                           title="loss", label="eval_Grad_loss")

        plt.savefig(os.path.join(self.logger.log_dir, 'train_loss_process.svg'))

        # matplotlib绘制训练过程图
        plt.figure(plot_channel + 1, figsize=(20, 10))
        plt.clf()
        # plt.ion()  # interactive mode on

        # 三种error
        temp_t = np.array(self.process_info["train_field_errors"], dtype=np.float32).transpose((1, 0, 2))
        temp_e = np.array(self.process_info["eval_field_errors"], dtype=np.float32).transpose((1, 0, 2))
        plt.subplot(231)
        plt.xscale('linear')
        plt.yscale('log')

        for index, (score_t, score_e) in enumerate(zip(temp_t, temp_e)):
            self.logger.scalar(self.process_info["iteration"], score_t[:, 0], title="L1_error",
                               label=self.field_name[index] + "_train")
            self.logger.scalar(self.process_info["iteration"], score_e[:, 0], title="L1_error",
                               label=self.field_name[index] + "_eval")

        plt.subplot(232)
        plt.xscale('linear')
        plt.yscale('log')
        for index, (score_t, score_e) in enumerate(zip(temp_t, temp_e)):
            self.logger.scalar(self.process_info["iteration"], score_t[:, 1], title="L2_error",
                               label=self.field_name[index] + "_train")
            self.logger.scalar(self.process_info["iteration"], score_e[:, 1], title="L2_error",
                               label=self.field_name[index] + "_eval")

        plt.subplot(233)
        plt.xscale('linear')
        plt.yscale('log')
        for index, (score_t, score_e) in enumerate(zip(temp_t, temp_e)):
            self.logger.scalar(self.process_info["iteration"], score_t[:, 2], title="Max_error",
                               label=self.field_name[index] + "_train")
            self.logger.scalar(self.process_info["iteration"], score_e[:, 2], title="Max_error",
                               label=self.field_name[index] + "_eval")

        train_scores = np.array(self.process_info["train_result_scores"], dtype=np.float32)
        eval_scores = np.array(self.process_info["eval_result_scores"], dtype=np.float32)

        plt.subplot(234)
        plt.xscale('linear')
        plt.yscale('linear')
        self.logger.scalar(self.process_info["iteration"], train_scores[:, 0], title="Nu_score", label="train")
        self.logger.scalar(self.process_info["iteration"], eval_scores[:, 0], title="Nu_score", label="eval")

        plt.subplot(235)
        plt.xscale('linear')
        plt.yscale('linear')
        self.logger.scalar(self.process_info["iteration"], train_scores[:, 1], title="f_score", label="train")
        self.logger.scalar(self.process_info["iteration"], eval_scores[:, 1], title="f_score", label="eval")

        plt.subplot(236)
        plt.xscale('linear')
        plt.yscale('linear')
        self.logger.scalar(self.process_info["iteration"], train_scores[:, 2], title="Tb_score", label="train")
        self.logger.scalar(self.process_info["iteration"], eval_scores[:, 2], title="Tb_score", label="eval")

        plt.savefig(os.path.join(self.logger.log_dir, 'train_error_process.svg'))

        result_t = self.process_info["train_flow_results"][epoch]
        result_e = self.process_info["eval_flow_results"][epoch]

        # matplotlib绘制Nu
        plt.figure(plot_channel + 2, figsize=(20, 10))
        plt.clf()
        # plt.ion()  # interactive mode on

        plt.subplot(231)
        self.logger.plot_regression([result_t[0][:, 0], result_t[1][:, 0]],
                                    title="train_Nu scores: " + str(train_scores[epoch, 0]))

        plt.subplot(234)
        self.logger.plot_regression([result_e[0][:, 0], result_e[1][:, 0]],
                                    title="eval_Nu scores: " + str(eval_scores[epoch, 0]))

        plt.subplot(232)
        self.logger.plot_regression([result_t[0][:, 1], result_t[1][:, 1]],
                                    title="train_Fa scores: " + str(train_scores[epoch, 1]))

        plt.subplot(235)
        self.logger.plot_regression([result_e[0][:, 1], result_e[1][:, 1]],
                                    title="eval_Fa scores: " + str(eval_scores[epoch, 1]))

        plt.subplot(233)
        self.logger.plot_regression([result_t[0][:, 2], result_t[1][:, 2]],
                                    title="train_Tb scores: " + str(train_scores[epoch, 2]))

        plt.subplot(236)
        self.logger.plot_regression([result_e[0][:, 2], result_e[1][:, 2]],
                                    title="eval_Tb scores: " + str(eval_scores[epoch, 2]))

        plt.savefig(os.path.join(self.logger.log_dir, 'train_Nu_f_Tb_process.svg'))




    def tensorboard_visual(self, epoch):

        # tensorboard 记录训练过程

        self.writer.scalars(names=('train', 'eval'),
                            values=(self.process_info["train_loss_logs"][epoch][0],
                                    self.process_info["eval_loss_logs"][epoch][0],),
                            iter=epoch, title='total loss')

        self.writer.scalars(names=('train', 'eval'),
                            values=(self.process_info["train_loss_logs"][epoch][1],
                                    self.process_info["eval_loss_logs"][epoch][1],),
                            iter=epoch, title='field loss')

        self.writer.scalars(names=('train', 'eval'),
                            values=(self.process_info["train_loss_logs"][epoch][2],
                                    self.process_info["eval_loss_logs"][epoch][2],),
                            iter=epoch, title='grad loss')



        # tensorboard 记录权值参数分布
        self.writer.histogram(self.G_net, epoch)

        # tensorboard 记录某batch的预测结果
        # self.writer.image(image=visual_images[:12].cpu().numpy(), iter=epoch,
        #                   labels=visual_labels, preds=visual_preds)

        # tensorboard 绘制提取特征的降维可视化结果
        # self.writer.embedding(features_reduce, iter=epoch, labels=features_labels)



    def output_field(self, names,grids, trues, preds, epoch):
        import pandas as pd
        import h5py

        N_f = self.input_size[1]
        N_x = self.input_size[2]
        N_y = self.input_size[3]
        N_n = N_x * N_y

        datamat = h5py.File(".//Re_100-1000_data//element.mat")
        Face = np.transpose(datamat["Element"], (1, 0))

        field_bound = self.field_bound[self.field_index, :]

        for (name, grid, true, pred) in zip(names, grids, trues, preds):

            name = str(name)

            grid = np.reshape(grid, (2, N_n)).transpose((1, 0))

            pred = np.reshape(pred, (N_f, N_n)).transpose((1, 0)) * (
                    field_bound[:, 1] - field_bound[:, 0]) + field_bound[:, 0]
            true = np.reshape(true, (N_f, N_n)).transpose((1, 0)) * (
                    field_bound[:, 1] - field_bound[:, 0]) + field_bound[:, 0]

            field = np.stack((true, pred), axis=-1).reshape((N_n, N_f * 2))

            data_out = np.concatenate((grid, field), axis=1)

            d1 = pd.DataFrame(data_out)
            d2 = pd.DataFrame(Face)

            output_file = self.predict_path + "//" + name + "_" + str(epoch) + ".dat"

            f = open(output_file, "w")
            f.write("%s\n" % ('TITLE = ' + '"' + name + '.data"'))
            f.write("%s" % ('VARIABLES = "X [ m ]","Y [ m ]"'))

            for index in range(len(self.field_name)):
                f.write("%s" % ',"' + self.field_name[index] + '_real", ' + self.field_name[index] + '_pred"')
            f.write("%s\n" % ',')

            f.write("%s\n" % ('ZONE T="symmetry 1", N=' + str(N_x * N_y) + ', E=' + str(
                (N_x - 1) * (N_y - 1)) + ', F=FEPOINT, ET=QUADRILATERAL'))
            f.close()

            d1.to_csv(output_file, index=False, mode='a', sep=" ", header=False)
            d2.to_csv(output_file, index=False, mode='a', sep=" ", header=False)


    def set_learning_rate(self, initial_learning_rate, boundary_epochs, rates_decay):

        vals = [initial_learning_rate * decay for decay in rates_decay]

        def learning_rate_fn(epoch):
            lt = [epoch < b for b in boundary_epochs] + [True]
            i = np.argmax(lt)
            return vals[i]

        return learning_rate_fn


    def set_learning_rate_exp(self, initial_learning_rate, end_epoch, rates_decay=0.75):

        lr = initial_learning_rate
        ep = end_epoch
        rd = rates_decay

        def learning_rate_fn(epoch):
            learning_rate = lr / math.pow((1 + 10 * epoch / ep), rd)
            return learning_rate

        return learning_rate_fn