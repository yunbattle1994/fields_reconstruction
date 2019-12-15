import numpy as np
import torch
import h5py


def caculate(name, bound, fields, grids, heflxs, datas):

    results = []

    for (field, grid, heflx, data) in zip(fields, grids, heflxs, datas):
        result = cal_all(data, heflx, grid, field, bound, name)
        results.append(result)

    return results


def get_parameters_of_nano(per):
    lamda_water = 0.597
    Cp_water = 4182.
    rho_water = 998.2
    miu_water = 9.93e-4

    lamda_al2o3 = 36.
    Cp_al2o3 = 773.
    rho_al2o3 = 3880.

    rho = per * rho_al2o3 + (1. - per) * rho_water
    Cp = ((1. - per) * rho_water * Cp_water + per * rho_al2o3 * Cp_al2o3) / rho
    miu = miu_water * (123. * per ** 2. + 7.3 * per + 1)
    DELTA = ((3. * per - 1.) * lamda_al2o3 + (2. - 3. * per) * lamda_water) ** 2
    DELTA = DELTA + 8. * lamda_al2o3 * lamda_water
    lamda = 0.25 * ((3 * per - 1) * lamda_al2o3 + (2 - 3 * per) * lamda_water + np.sqrt(DELTA))
    
    return float(lamda), float(Cp), float(rho), float(miu)


def cal_q_s(X, Y, T):
    index = np.zeros(792, dtype=np.bool)
    index[120:672] = True

    index1 = index.copy()
    index1[119] = True
    index1[671] = False

    index2 = index.copy()
    index2[120] = False
    index2[672] = True

    dx1 = X[index, 0]
    temp = -X[index1, 0]
    dx1 = dx1+temp
    dx2 = X[index, 0]
    temp = -X[index2, 0]
    dx2 = dx2 + temp
    dy1 = Y[index, 0] - Y[index1, 0]
    dy2 = Y[index, 0] - Y[index2, 0]
    ds1 = np.sqrt(dx1 ** 2 + dy1 ** 2)
    ds2 = np.sqrt(dx2 ** 2 + dy2 ** 2)
    ds = (ds1 + ds2) / 2
    dT = T[index, 0] - T[index, 1]
    dx = X[index, 1] - X[index, 0]
    dy = Y[index, 1] - Y[index, 0]

    dn_g = np.array([dx, dy], dtype=np.float32)
    norm_dng = np.sqrt(dx ** 2 + dy ** 2)
    dn_g = dn_g / norm_dng
    dn_r = np.array([dx1 / ds1, dy1 / ds1], dtype=np.float32) - np.array([dx2 / ds2, dy2 / ds2], dtype=np.float32)
    norm_dnr = dn_r[0, :] ** 2 + dn_r[1, :] ** 2
    norm_dnr = np.sqrt(norm_dnr)
    dn_r = dn_r / norm_dnr
    dn_r = np.array([dn_r[1, :], -dn_r[0, :]], dtype=np.float32)
    ind = (dn_r[1, :] > 0)
    dn_r[:, ind] = - dn_r[:, ind]

    cos_ = np.dot(dn_r.T, dn_g)
    cos = np.diag(cos_)

    dTdy = -dT / dy
    grad = dTdy / cos

    dq = grad * ds
    q = np.sum(dq)
    s = np.sum(ds)

    return q, s


def cal_f(X, Y, P):

    # F_inn = (np.power(10, P[119, 1:]) + np.power(10, P[119, 0:-1])) / 2 - 1
    # F_out = (np.power(10, P[672, 1:]) + np.power(10, P[672, 0:-1])) / 2 - 1

    F_inn = (P[119, 1:] + P[119, 0:-1]) / 2
    F_out = (P[672, 1:] + P[672, 0:-1]) / 2

    dy_inn = Y[119, 1:40] - Y[119, 0:39]
    dy_out = Y[672, 1:40] - Y[672, 0:39]

    D_P = np.sum(F_inn * dy_inn) / np.sum(dy_inn) - np.sum(F_out * dy_out) / np.sum(dy_out)

    return D_P

def cal_tb(X, Y, T):


    F_T = T[119:673,:]
    # F_X = X[119:673,:]
    # F_Y = Y[119:673,:]

    dxx = X[119:672, :] - X[120:673, :]
    dxy = Y[119:672, :] - Y[120:673, :]
    dyx = X[119:673, 1:40] - X[119:673, 0:39]
    dyy = Y[119:673, 1:40] - Y[119:673, 0:39]

    ds1 = np.abs(np.multiply(dxx[:, :-1], dyy[1:]) - np.multiply(dxy[:, :-1], dyx[1:])) / 2
    ds2 = np.abs(np.multiply(dxx[:, 1:], dyy[:-1]) - np.multiply(dxy[:, 1:], dyx[:-1])) / 2
    ds = ds1 + ds2

    M_T = (F_T[1:, 1:] + F_T[1:, :-1] + F_T[:-1, 1:] + F_T[:-1, :-1]) / 4

    Tb = np.sum(np.multiply(ds, M_T))/np.sum(ds)

    return Tb


def cal_tw(X, Y, T):

    up_t = T[119:673, 39]
    down_t = T[119:673, 0]

    temp = np.sqrt((X[119:672, 39]-X[120:673, 39])**2 + (Y[119:672,39]-Y[120:673,39])**2)
    up_dl = np.array([temp[0] / 2] + (np.array(temp[0:552] + temp[1:553]) / 2).tolist() + [temp[552] / 2], dtype=np.float32)
    temp = np.sqrt((X[119:672, 0] - X[120:673, 0])**2 + (Y[119:672, 0] - Y[120:673, 0])**2)
    down_dl = np.array([temp[0] / 2] + (np.array(temp[0:552] + temp[1:553]) / 2).tolist() + [temp[552] / 2], dtype=np.float32)
    Tw = (np.dot(up_t, up_dl)+np.dot(down_t, down_dl)) / np.sum(up_dl+down_dl)

    return Tw


def cal_all(data, q, grid, field, boundary, names=('T', 'P')):

    D = float(2 * 25 * 1e-6)
    L = float(350 * 1e-6)

    per = data[3]
    Re = data[0]
    lamda, Cp, rho, miu = get_parameters_of_nano(per)

    X = grid[:, :, 0].squeeze()
    Y = grid[:, :, 1].squeeze()

    Nu = float(1.)
    f = float(1.)
    Tb = float(1.)

    for name in names:

        if name == "T" or "t":
            Th = boundary[1, 1]
            Tc = boundary[1, 0]
            T = field[1, :, :] * (Th - Tc) + Tc

            Tw = cal_tw(X, Y, T)
            Tb = cal_tb(X, Y, T)
            h = q/np.abs(Tw-Tb)
            Nu = h * D / lamda
    #
        if name == "P" or "p":
            Pi = boundary[0, 1]
            Po = boundary[0, 0]
            P = field[0, :, :] * (Pi - Po) + Po

            vel = Re * miu / rho / D
            Dp = cal_f(X, Y, P)
            f = Dp * D / 2 / L / vel / vel / rho

    return Nu, f, Tb



if __name__ == '__main__':
    # per = 0.01
    # lamda, Cp, rho, miu = get_parameters_of_nano(per)
    # X = np.random.rand(792, 40)
    # Y = np.random.rand(792, 40)
    # T = np.random.rand(792, 40)
    path = 'G:\\lty\\Field_reconstruct\\Re_exp_data\\exp_DG_single_10-1000.mat'
    datamat = h5py.File(path)
    # X = np.transpose(datamat['X'], (1, 0))
    # Y = np.transpose(datamat['Y'], (1, 0))
    # T = np.transpose(datamat['T'], (1, 0))
    # q, s = cal_q_s(X, Y, T)

    fields = np.transpose(datamat['field'], (3, 2, 1, 0))[:, :, :, :]
    grids = np.transpose(datamat['grids'], (3, 2, 1, 0))
    scalar = np.transpose(datamat['scalar'], (1, 0)).squeeze()
    design = np.transpose(datamat['data'], (1, 0)).squeeze()
    

    NU = scalar[:, -1]
    Tbs = scalar[:, 2]
    Q = scalar[:, 1]

    boundary = np.array([[0, 8e5], [292, 322], [-100, 300], [-50, 100]]).astype(np.float32)

    fields = (fields - boundary[:, 0]) / (boundary[:, 1] - boundary[:, 0])


    results = []

    import time
    s_time = time.time()


    for (field, grid, q, data, nu) in zip(fields, grids, Q, design, NU):

        grid = grid[:, :, :].squeeze()
        X = grid[:, :, 0].squeeze()
        Y = grid[:, :, 1].squeeze()
        F = field.transpose([2, 0, 1])

        result = cal_all(data, q, grid, F, boundary, )
        results.append(result)

    e_time = time.time()
    print(e_time-s_time)

    import matplotlib.pyplot as plt
    import Utilize.data_visualization as visual

    logger = visual.matplotlib_vision("re\\")

    plt.figure(1, figsize=(20, 15))
    plt.clf()
    plt.ion()  # interactive mode on

    plt.subplot(121)
    logger.plot_regression([scalar[:, -1], np.array(results, dtype=np.float32)[:, 0]],)

    plt.subplot(122)
    plt.scatter(design[:, 0], np.array(results, dtype=np.float32)[:, 1])

    
    plt.pause(100)
