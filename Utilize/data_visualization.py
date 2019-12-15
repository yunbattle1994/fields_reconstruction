import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import seaborn as sbn
import pandas as pd
from mpl_toolkits.mplot3d.axes3d import Axes3D
from tensorboardX import SummaryWriter




class matplotlib_vision(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.log_dir = log_dir
        sbn.set_style('ticks')

    def scalar(self, x, y, label='train', title='Accuracy each epoch', ax=None):
        sbn.set_style('ticks')
        sbn.set(color_codes=True)
        plt.title(title)
        plt.plot(x, y, label=label)
        plt.legend()
        plt.grid(True)  # 添加网格
        # plt.pause(0.001)

    # plot confuse matrix
    def confusion_matrix(self, confuse_matrix, labels=None, title='Confuse Matrix', cmap=plt.get_cmap('Reds')):

        sbn.set_style('ticks')
        sbn.set(color_codes=True)

        num_classes = confuse_matrix.shape[0]

        sbn.heatmap(confuse_matrix, cmap=cmap, annot=True, fmt="5d", cbar=True)

        plt.title(title)

        xlocation = np.array(range(num_classes))
        if labels == None:
            plt.xticks(xlocation, range(num_classes))
            plt.yticks(xlocation, range(num_classes))
        else:
            plt.xticks(xlocation, labels, rotation=90)
            plt.yticks(xlocation, labels)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # plt.pause(0.001)


    def image(self, images, labels, preds, size=(3, 4)):

        """
            images is a tensor [batch, channel, width, height]
        """
        sbn.set_style('ticks')
        sbn.set(color_codes=True)

        total = size[0] * size[1]

        if len(images.shape) == 4:   #

            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)

            for index, image in enumerate(images[:total]):

                if image.shape[0] == 1:
                    img = np.squeeze(image)  # transforms.ToPILImage(image)
                    plt.subplot(size[0], size[1], index + 1)
                    plt.imshow(img, cmap='gray')
                else:
                    for channel, _ in enumerate(image):
                        image[channel] *= std[channel]
                        image[channel] += mean[channel]
                        image[channel] *= 255

                    img = image.transpose(1, 2, 0)  # transforms.ToPILImage(image)
                    plt.subplot(size[0], size[1], index + 1)
                    plt.imshow(img)

                title = "label: " + str(labels[index]) + "  pred: " + str(preds[index])
                plt.title(title)

        elif len(images.shape) == 3:

            for index, signal in enumerate(images[:total]):

                t = np.arange(0, signal.shape[1])
                plt.subplot(size[0], size[1], index + 1)

                for channel, _ in enumerate(signal):

                    plt.plot(t, signal[channel, :])

                title = "label: " + str(labels[index]) + "  pred: " + str(preds[index])
                plt.title(title)

        # plt.pause(0.001)

    def fields_compare(self, true, pred, size=(2, 4)):
        sbn.set_style('ticks')
        sbn.set(color_codes=True)
        total = size[0] * size[1]

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        for index, image in enumerate(true[:total]):

            if image.shape[0] == 1:
                img_true = np.squeeze(image)  # transforms.ToPILImage(image)
                img_pred = np.squeeze(pred[index])
                img = np.hstack((img_true, img_pred))

                plt.subplot(size[0], size[1], index + 1)
                plt.imshow(img[200:600, :]*255, cmap='gray')
            else:
                for channel, _ in enumerate(image):
                    image[channel] *= std[channel]
                    image[channel] += mean[channel]
                    image[channel] *= 255

                img = image.transpose(1, 2, 0)  # transforms.ToPILImage(image)
                plt.subplot(size[0], size[1], index + 1)
                plt.imshow(img)

                title = "true " + "    pred"
                plt.title(title)

        # plt.pause(0.001)


    def features_output(self, features, visual_channel=10, save_path=".//"):
        sbn.set_style('ticks')
        sbn.set(color_codes=True)
        if features.ndim == 1:
            features = features[:, np.newaxis].transpose(1, 0)
            plt.imshow(features, cmap='RdBu', aspect=0.1)
            plt.axis('off')
            plt.savefig(save_path + '\\channels_' + str(0) + '.jpg', dpi=300)

        else:

            visual_channel = features.shape[0]
            for channel in range(visual_channel):    #取第一列数据，即第一个图片特征数据
                # 仅输出第一个图片的特征图片形式，为1channel的灰度
                feature = features[channel, :, :].transpose(1, 0)
                # 将数据转化为0-1之间
                feature = 1.0 / (1 + np.exp(-1 * feature))
                # 将数据转化为0-255之间整数
                feature = np.round(feature * 255)
                # 图片保存
                # cv2.imwrite(save_path + 'layers_' + layer + '_channels_' + str(channel) + '.jpg', feature)
                plt.imshow(feature, cmap='jet')
                plt.axis('off')

                plt.savefig(save_path + '\\channels_' + str(channel) + '.jpg', dpi=100)

                # # plt.pause(0.001)


    def geoms_compare(self, true, pred, fig, size=(2, 4)):

        sbn.set_style('ticks')
        sbn.set(color_codes=True)

        total = size[0] * size[1]
        n_groups = true.shape[1]

        for i, true_value in enumerate(true[:total]):

            ax = fig.add_subplot(size[0], size[1], i+1)

            index = np.arange(n_groups)
            bar_width = 0.35

            opacity = 0.4

            ax.bar(index, true_value, bar_width,
                            alpha=opacity, color='b', label='true')

            ax.bar(index + bar_width, pred[i, :], bar_width,
                            alpha=opacity, color='r',label='pred')

            # ax.set_xlabel('Group')
            ax.set_ylabel('Scores')
            # ax.set_title('Scores by group and gender')
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels((''))
            ax.legend()
            fig.tight_layout()

        # plt.pause(0.001)



    def embedding_3d(self, reduce_data, labels, axes, title=None):
        sbn.set_style('ticks')
        sbn.set(color_codes=True)
        #坐标缩放到[0,1]区间
        x_min, x_max = np.min(reduce_data, axis=0), np.max(reduce_data, axis=0)
        reduce_data = (reduce_data - x_min) / (x_max - x_min)
        #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits

        for i in range(reduce_data.shape[0]):
            axes .text(reduce_data[i, 0], reduce_data[i, 1], reduce_data[i,2],str(labels[i]),
                     color=plt.cm.Set1(labels[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})

        # plt.pause(0.001)
        if title is not None:
            plt.title(title)


    def embedding_2d(self, reduce_data, labels, title=None):
        sbn.set_style('ticks')
        sbn.set(color_codes=True)
        #坐标缩放到[0,1]区间
        x_min, x_max = np.min(reduce_data, axis=0), np.max(reduce_data, axis=0)
        reduce_data = (reduce_data - x_min) / (x_max - x_min)
        #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits

        # 可视化
        plt.scatter(reduce_data[:, 0], reduce_data[:, 1], c=labels,
                    edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('Spectral', 10))
        plt.colorbar(label='digit label', ticks=range(10))
        plt.clim(-0.5, 9.5)
        # plt.pause(0.001)
        if title is not None:
            plt.title(title)


    def plot_regression(self, results, axis=0, title=None):
        # 所有功率预测误差与真实结果的回归直线
        sbn.set_style('ticks')
        sbn.set(color_codes=True)
        if np.ndim(results[0]) > 1:
            true = results[0][:, axis]
            pred = results[1][:, axis]
        else:
            true = results[0]
            pred = results[1]

        relative_err = (pred - true) / true

        max_value = max(true) # math.ceil(max(true)/100)*100
        min_value = min(true) # math.floor(min(true)/100)*100
        split_value = np.linspace(min_value, max_value, 11)

        split_dict = {}
        split_label = np.zeros(len(true), np.int)
        for i in range(len(split_value)):
            split_dict[i] = str(split_value[i])
            index = true >= split_value[i]
            split_label[index] = i + 1

        result = pd.DataFrame({'truth': true, 'preds': pred, 'label': split_label, 'error': relative_err})
        sbn.regplot(x='truth', y='preds', data=result)
        # sns.jointplot(x="truth", y="preds", data=result, kind='hex')
        # sns.boxplot(x='label', y='error', data=result, whis=10)
        plt.ylim((min_value, max_value))
        plt.xlim((min_value, max_value))
        plt.ylabel('pred health')
        plt.xlabel('real health')
        plt.grid(True)  # 添加网格
        # plt.ylim((-0.2, 0.2))
        # plt.pause(0.001)


class tensorboard_vision(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir, comment='net_model')

    def graph(self, net_model, x):
        self.writer.add_graph(net_model, x, False)


    def scalar(self, value, iter, name='loss'):
        self.writer.add_scalar(name, value, iter)

    def scalars(self, values, iter, names, title):
        for name, value in zip(names, values):
            self.writer.add_scalars(title, {name: value},  iter)

    def image(self, image, iter, labels, preds):

        total = image.shape[0]
        for index in range(total):
            name = ("label: " + str(labels[index]) + "  pred: " + str(preds[index]))
            self.writer.add_image(name, image[index], iter)

    def histogram(self, net_model, iter):
        for name, param in net_model.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), iter)

    def embedding(self, features, iter, labels):
        self.writer.add_embedding(features, labels, global_step=iter)

if __name__ == '__main__':
    import torch


    # nz = 7
    # true = torch.randn(20, nz).numpy()
    # pred = torch.randn(20, nz).numpy()
    #
    logger = matplotlib_vision("//")
    #
    # logger.geoms_compare(true, pred)

    true = torch.randn(20, 1, 792, 40).numpy()
    pred = torch.randn(20, 1, 792, 40).numpy()

    plt.figure(1, figsize=(15, 15))
    plt.clf()
    plt.ion()

    logger.fields_compare(true, pred)

