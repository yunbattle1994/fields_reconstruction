
import numpy as np
import os
import h5py

def get_local_information(fields, points, layout='B'):
    # points
    # inlet extension, short1, dim1, short2, dim2, short3, outlet extension
    # 1-120, 120-248, 248-327, 327-466, 466-545, 545-673, 673-792
    grid_number = [1, 120, 248, 327, 466, 545, 673, 792]
    inlets = fields[:, 0, :, :2]
    outlets = fields[:, -1, :, :2]
    ups = fields[:, :, 0, :2]
    downs = fields[:, :, -1, :2]
    # 固定的有：入口，出口
    info_ins = np.mean(inlets, axis=1)
    info_ous = np.mean(outlets, axis=1)
    info_ups = []
    info_dws = []
    for i, point in enumerate(points):
        if point != 0:
            locates = np.linspace(grid_number[i], grid_number[i + 1], num=point+1, endpoint=False, dtype=int)[1:]
            if info_ups == []:
                info_ups = ups[:, locates, :]
                info_dws = ups[:, locates, :]
            else:
                info_ups = np.concatenate((info_ups, ups[:, locates, :]), axis=1)
                info_dws = np.concatenate((info_dws, ups[:, locates, :]), axis=1)
    if layout == 'B':
        exp_pt = np.concatenate((info_ins[:, np.newaxis,:], info_ups, info_dws, info_ous[:, np.newaxis,:]), axis=1)
    elif layout == 'D':
        exp_pt = np.concatenate((info_ins[:, np.newaxis,:], info_dws, info_ous[:, np.newaxis,:]), axis=1)
    elif layout == 'U':
        exp_pt = np.concatenate((info_ins[:, np.newaxis,:], info_ups, info_ous[:, np.newaxis,:]), axis=1)

    return exp_pt

if __name__ ==  '__main__':
    file_name = []
    inputs = []  # 存储测点数据
    fields = []  # 流场数据
    grids = []  # 网格数据
    scalar = []  # 统计数据
    design = []  # 设计变量

    # 修改
    input_dir = '..\\dim_pro8_data'
    matname = 'try'
    path = input_dir + "\\dim_pro8_single_" + matname + ".mat"
    datamat = h5py.File(path)
    # 修改

    # self.inputs = np.transpose(datamat['exp_pt'], (2, 1, 0))[str_size:end_size, :, points]
    fields = np.transpose(datamat['field'], (3, 2, 1, 0))[:, :, :, :]
    points = [1, 0, 1, 1, 1, 1, 1]
    layout = 'D'
    exp_pt = get_local_information(fields=fields, points=points, layout=layout)
    a=0