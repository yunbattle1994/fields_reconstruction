from sklearn.model_selection import train_test_split
import numpy as np
import os
import h5py
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time


from Utilize import data_torch as datatorch, thermal_evaluation_gpu as thermal


def get_DG_loaders(std=0, train_size=1.0, batch_size_train=32, batch_size_eval=64, limit_size=0.5,
                   input_dir='data', field_name=("U",), field_index=(0,),
                   field_bound=np.array([[-1, 2.1e7], [292, 322], [-100, 300], [-50, 100]]),
                   points=np.array([2, 2, 2, 2, 2, 2, 2]), layout='B', if_design=False,if_grid=False, matname="all"):
    # 数据增强
    if std > 0:
        transform_input = transforms.Compose(  # 定义一套torch格式下的图片编辑动作
            [
                datatorch.Add_noise(),
                datatorch.ToTensor_1D(),
            ]
        )
    else:

        transform_input = None

    transform_field = transforms.Compose(  # 定义一套torch格式下的图片编辑动作
        [
            transforms.ToTensor(),
        ]
    )
    # 数据读取
    start_time = time.time()
    DG_data = DimpleGroove_data(input_dir=input_dir)
    DG_data.load_data(field_index=field_index, field_bound=field_bound, points=points, layout=layout, if_design=if_design, if_grid=if_grid, matname=matname)
    print("Load Dimple_Groove Data Time {:.4f} ".format(time.time() - start_time))

    # 数据划分
    start_time = time.time()
    data_t, data_e = DG_data.split_data(train_size=train_size, limit_size=limit_size)

    if std > 0:
        data_e.inputs = data_e.inputs * \
                        (1. + np.random.randn(data_e.inputs.shape[0], data_e.inputs.shape[1],).astype(np.float32) * std)

    print("Split Dimple_Groove Data Time {:.4f} ".format(time.time() - start_time))

    # 计算流场统计结果
    start_time = time.time()
    data_t.flowres = thermal.cal_all_numpy(field_name, field_bound, data_t.fields.transpose([0, 3, 1, 2]),
                                      data_t.grids.transpose([0, 3, 1, 2]), data_t.scalar[:, 1], data_t.design)

    data_e.flowres = thermal.cal_all_numpy(field_name, field_bound, data_e.fields.transpose([0, 3, 1, 2]),
                                      data_e.grids.transpose([0, 3, 1, 2]), data_e.scalar[:, 1], data_e.design)

    print("Caculate flow field result {:.4f} ".format(time.time() - start_time))


    # pytorch训练集,测试集
    # 修改
    start_time = time.time()
    dataset_t = datatorch.Dataset_dimplegroove(data_t, field_transform=transform_field, target_transform=transform_input)
    loader_t = DataLoader(dataset_t, batch_size=batch_size_train, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

    dataset_e = datatorch.Dataset_dimplegroove(data_e, field_transform=transform_field, target_transform=None)
    loader_e = DataLoader(dataset_e, batch_size=batch_size_eval, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    print("Pytorch data loader {:.4f} ".format(time.time() - start_time))


    return loader_t, loader_e



def get_DG_sklearn(std=0, train_size=0.5,
                   input_dir='data', field_name="U", field_index=(0,),
                   field_bound=np.array([[-1, 2.1e7], [292, 322], [-100, 300], [-50, 100]]),
                   points=np.array([2, 2, 2, 2, 2, 2, 2]), layout='B', if_design=False, if_grid=False):

    # 数据读取
    start_time = time.time()
    DG_data = DimpleGroove_data(input_dir=input_dir)
    DG_data.load_data(field_index=field_index, field_bound=field_bound, points=points, layout=layout,
                      if_grid=if_grid, if_design=if_design)
    print("Load Dimple_Groove Data Time {:.4f} ".format(time.time() - start_time))

    # 数据划分
    start_time = time.time()
    data_t, data_e = DG_data.split_data(train_size=train_size)

    if std > 0:
        data_e.inputs = data_e.inputs * \
                        (1. + np.random.randn(data_e.inputs.shape[0], data_e.inputs.shape[1],).astype(np.float32) * std)

    print("Split Dimple_Groove Data Time {:.4f} ".format(time.time() - start_time))

    # 计算流场统计结果
    start_time = time.time()
    data_t.flowres = thermal.cal_all_numpy(field_name, field_bound, data_t.fields.transpose([0, 3, 1, 2]),
                                      data_t.grids.transpose([0, 3, 1, 2]), data_t.scalar[:, 1], data_t.design)

    data_e.flowres = thermal.cal_all_numpy(field_name, field_bound, data_e.fields.transpose([0, 3, 1, 2]),
                                      data_e.grids.transpose([0, 3, 1, 2]), data_e.scalar[:, 1], data_e.design)

    print("Caculate flow field result {:.4f} ".format(time.time() - start_time))


    end_time = time.time()

    print("Load Dimple_Groove Data Time {:.4f} ".format(end_time-start_time))

    return data_t, data_e


class DG_data:
    pass


class DimpleGroove_data(object):

    def __init__(self, input_dir='data'):
        # 文件读取路径
        self.input_dir = input_dir

    def load_data(self,  field_index=(1,), field_bound=np.array([[-1, 2.1e7], [292, 322], [-100, 300], [-50, 100]]),
                  points=np.array([2, 2, 2, 2, 2, 2, 2]), layout='B', if_design=False, if_grid=False, matname="all"):  # 用于读取文件

        # self.input_dir = input_dir
        self.file_name = []
        self.inputs = []  # 存储测点数据
        self.fields = []  # 流场数据
        self.grids = []  # 网格数据
        self.scalar = []  # 统计数据
        self.design = []  # 设计变量


        all_path = []  # 所有文件的路径

        for root, sub_folder, file_names in os.walk(self.input_dir):
            for file_name in file_names:
                if file_name.endswith('.mat') or file_name.endswith('.MAT'):
                    filepath = os.path.join(root, file_name)
                    all_path.append(filepath)  # 信号地址
                    self.file_name.append(filepath)
        # input_data = pd.DataFrame({"path": case_path, "label": case_label})
        # input_data.to_csv(os.path.join(self.save_dir, self.choose_load + '.csv'))

        # 修改
        path = self.input_dir + "\\dim_pro8_single_" + matname + ".mat"
        datamat = h5py.File(path)
        # 修改

        # self.inputs = np.transpose(datamat['exp_pt'], (2, 1, 0))[str_size:end_size, :, points]
        self.fields = np.transpose(datamat['field'], (3, 2, 1, 0))[:, :, :, field_index]
        self.grids = np.transpose(datamat['grids'], (3, 2, 1, 0))[:, :, :, :2]
        self.scalar = np.transpose(datamat['scalar'], (1, 0)).squeeze()[:, :]
        self.design = np.transpose(datamat['data'], (1, 0)).squeeze()[:, :4]



        # 数据归一化到[0 1]
        self.field_bound = field_bound.astype(np.float32)
        self.fields = (self.fields - self.field_bound[field_index, 0]) \
                      / (self.field_bound[field_index, 1] - self.field_bound[field_index, 0])
        if if_grid:
            # self.grid_bound = np.array([[0, 5e-4], [0, 5e-5]]).astype(np.float32)# Re_100-1000
            self.grid_bound = np.array([[0, 5e-3], [0, 5e-4]]).astype(np.float32)  # Re_100-1000
            self.grids = (self.grids - self.grid_bound[:, 0]) \
                          / (self.grid_bound[:, 1] - self.grid_bound[:, 0])


        if if_design:

            path = self.input_dir + "\\dim_pro8_single_" + "all" + ".mat"
            datamat_all = h5py.File(path)
            design_all = np.transpose(datamat_all['data'], (1, 0)).squeeze()[:, :4]

            Max_bound = np.max(design_all, axis=0) # np.array((1000, 294, 1e5, 0.01,  70, 70, 28, 28, 250, 275), dtype=np.float32)
            Min_bound = np.min(design_all, axis=0) # np.array((10, 293, 1e4,  0.001, 10, 10, -15, -15, 25., 15), dtype=np.float32)

            delta = (Max_bound - Min_bound)
            for i, d in enumerate(delta):
                if d == 0:
                    delta[i] = 1
            self.inputs = (self.design - Min_bound) / delta
        else:
            self.inputs = self.get_local_information(points=points, layout=layout)
            # self.inputs[:, 0, :] = (self.inputs[:, 0, :] - field_bound[0, 0]) / (field_bound[0, 1] - field_bound[0, 0])
            # self.inputs[:, 1, :] = (self.inputs[:, 1, :] - field_bound[1, 0]) / (field_bound[1, 1] - field_bound[1, 0])

            size = np.shape(self.inputs)
            self.inputs = np.reshape(self.inputs, (size[0], size[1]*size[2]))

        #

    def split_data(self, train_size, limit_size=0.5):

        t = DG_data()
        e = DG_data()
        # 修改
        # 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)
        if limit_size >= 1.0:
            limit_size = 0.99999

        scalar_, e.scalar, design_, e.design, inputs_, e.inputs, grids_, e.grids, fields_, e.fields = train_test_split(
            self.scalar, self.design, self.inputs, self.grids, self.fields, train_size=limit_size, random_state=0)

        if train_size >= 1.0:
            train_size = 0.99999

        t.scalar, _, t.design, _,  t.inputs, _, t.grids, _, t.fields, _, = train_test_split(
            scalar_, design_, inputs_, grids_, fields_, train_size=train_size, random_state=0)

        return t, e

    def get_local_information(self, points, layout='B'):
        # points
        # inlet extension, short1, dim1, short2, dim2, short3, outlet extension
        # 1-120, 120-248, 248-327, 327-466, 466-545, 545-673, 673-792
        grid_number = [1, 120, 248, 327, 466, 545, 673, 792]
        inlets = self.fields[:, 0, :, :2]
        outlets = self.fields[:, -1, :, :2]
        downs = self.fields[:, :, 0, :2]
        ups = self.fields[:, :, -1, :2]
        # 固定的有：入口，出口
        info_ins = np.mean(inlets, axis=1)
        info_ous = np.mean(outlets, axis=1)
        info_ups = []
        info_dws = []
        for i, point in enumerate(points):
            if point != 0:
                locates = np.linspace(grid_number[i], grid_number[i + 1], num=point + 1, endpoint=False, dtype=int)[1:]
                if info_ups == []:
                    info_ups = ups[:, locates, :]
                    info_dws = downs[:, locates, :]
                else:
                    info_ups = np.concatenate((info_ups, ups[:, locates, :]), axis=1)
                    info_dws = np.concatenate((info_dws, downs[:, locates, :]), axis=1)
        if layout == 'B':
            exp_pt = np.concatenate((info_ins[:, np.newaxis, :], info_ups, info_dws, info_ous[:, np.newaxis, :]),
                                    axis=1)
        elif layout == 'D':
            exp_pt = np.concatenate((info_ins[:, np.newaxis, :], info_dws, info_ous[:, np.newaxis, :]), axis=1)
        elif layout == 'U':
            exp_pt = np.concatenate((info_ins[:, np.newaxis, :], info_ups, info_ous[:, np.newaxis, :]), axis=1)

        return exp_pt






if __name__ == '__main__':
    is_train = 1
    points = np.array([0, 1, 3, 5, 7, 9, 10, 12, 14, 16, 18, 19])
    points = np.linspace(0, 19, 20).astype(np.int)

    #
    t_loader, eval_loader = get_DG_loaders(std=0, train_size=1.0, batch_size_train=2, batch_size_eval=2, limit_size=1.0,
                                           input_dir='dim_pro8_data', field_name=("P","T", "U", "V"), field_index=(0, 1, 2, 3),
                                           field_bound=np.array([[-1, 2.1e7], [292, 322], [-100, 300], [-50, 100]]),
                                           points=points, if_design=True, matname="try")

    a = 0


