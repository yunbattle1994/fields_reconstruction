import torch
from torch.utils.data import Dataset
import numpy as np
from random import randint, random


# 训练集读取
class Dataset_dimplegroove(Dataset):
    def __init__(self, data, field_transform=None, target_transform=None):
        self.indexs = data.scalar[:, 0]
        self.fields = data.fields[:, :, :, :]
        self.inputs = data.inputs
        self.grids = data.grids
        self.heflxs = data.scalar[:, 1]
        self.design = data.design
        self.flowres = data.flowres

        self.field_transform = field_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = self.inputs[index]
        field = self.fields[index]
        name = self.indexs[index]
        grid = self.grids[index]
        heflx = self.heflxs[index]
        design = self.design[index]
        flowres = self.flowres[index]
        if self.field_transform is not None:
            field = self.field_transform(field)
            grid = self.field_transform(grid)
        if self.target_transform is not None:
            input = self.target_transform(input)
        return name, input, grid, field, heflx, design, flowres

    def __len__(self):
        return len(self.indexs)



# 训练集读取
class Dataset_foil(Dataset):
    def __init__(self, data, input_transform=None, target_transform=None):

        self.fields = data.fields
        self.grids = data.grids
        self.time_size = data.time_size
        self.sample_size = len(data.fields) - data.time_size
        self.target_field = data.field_index

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        out_index = index + self.time_size
        input = self.fields[index:out_index, :, :, :]
        input = np.transpose(input, (1, 2, 3, 0))
        shape = np.shape(input)
        input = np.resize(input, [shape[0], shape[1], shape[2]*shape[3]])

        output = self.fields[out_index, :, :, :]
        output = output[:, :, self.target_field]
        output = np.resize(output, [shape[0], shape[1], len(self.target_field)])

        grid = self.grids[out_index]

        if self.input_transform is not None:
            input = self.input_transform(input)
            grid = self.input_transform(grid)
        if self.target_transform is not None:
            output = self.target_transform(output)
        return out_index, input, output, grid

    def __len__(self):
        return self.sample_size





class TestDataSet(object):
    def __init__(self, filenames, imgs, transform=None, target_transform=None):
        self.filenames = filenames
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        filename = self.filenames[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.filenames)

class Add_noise(object):

    def __call__(self, x):
        std = np.random.uniform(-0.05, 0.05)
        r = np.random.randn(len(x),).astype(np.float32) * std
        y = x * (1. + r)
        return y

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToTensor_1D(object):
    def __call__(self, data):
        return torch.from_numpy(data)

    def __repr__(self):
        return self.__class__.__name__ + '()'