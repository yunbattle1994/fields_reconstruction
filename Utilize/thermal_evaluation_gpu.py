import numpy as np
import torch
import h5py



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
    lamda = 0.25 * ((3 * per - 1) * lamda_al2o3 + (2 - 3 * per) * lamda_water + torch.sqrt(DELTA))

    return lamda, Cp, rho, miu


def cal_q_s(X, Y, T, device=torch.device('cuda:' + str(0))):

    index = range(120, 672, 1)
    index1 = range(119, 671, 1)
    index2 = range(121, 673, 1)

    dx1 = X[index, 0]
    temp = -X[index1, 0]
    dx1 = dx1 + temp
    dx2 = X[index, 0]
    temp = -X[index2, 0]
    dx2 = dx2 + temp
    dy1 = Y[index, 0] - Y[index1, 0]
    dy2 = Y[index, 0] - Y[index2, 0]
    ds1 = torch.sqrt(dx1 ** 2 + dy1 ** 2)
    ds2 = torch.sqrt(dx2 ** 2 + dy2 ** 2)
    ds = (ds1 + ds2) / 2
    dT = T[index, 0] - T[index, 1]
    dx = X[index, 1] - X[index, 0]
    dy = Y[index, 1] - Y[index, 0]

    dn_g = torch.stack((dx, dy))
    norm_dng = torch.sqrt(dx ** 2 + dy ** 2)
    dn_g = dn_g / norm_dng
    dn_r = torch.stack((dx1 / ds1, dy1 / ds1)) - torch.stack((dx2 / ds2, dy2 / ds2))
    norm_dnr = torch.sqrt(dn_r[0, :] ** 2 + dn_r[1, :] ** 2)
    dn_r = dn_r / norm_dnr
    dn_r = torch.stack((dn_r[1, :], -dn_r[0, :]))
    ind = (dn_r[1, :] > 0)
    dn_r[:, ind] = - dn_r[:, ind]

    cos = torch.diag(torch.mm(torch.transpose(dn_r, 1, 0), dn_g))

    dTdy = -dT / dy
    grad = dTdy / cos

    dq = grad * ds
    q = torch.sum(dq)
    s = torch.sum(ds)

    return q, s


def cal_f(X, Y, P, device=torch.device('cuda:' + str(0))):
    # F_inn = (np.power(10, P[119, 1:]) + np.power(10, P[119, 0:-1])) / 2 - 1
    # F_out = (np.power(10, P[672, 1:]) + np.power(10, P[672, 0:-1])) / 2 - 1

    F_inn = (P[:, 119, 1:] + P[:, 119, 0:-1]) / 2
    F_out = (P[:, 672, 1:] + P[:, 672, 0:-1]) / 2

    dy_inn = Y[:, 119, 1:] - Y[:, 119, 0:-1]
    dy_out = Y[:, 672, 1:] - Y[:, 672, 0:-1]

    D_P = torch.sum(F_inn * dy_inn, dim=(1,)) / torch.sum(dy_inn, dim=(1,)) \
          - torch.sum(F_out * dy_out, dim=(1,)) / torch.sum(dy_out, dim=(1,))

    return D_P


def cal_tb(X, Y, T, device=torch.device('cuda:' + str(0))):
    F_T = T[:, 119:673, :]
    # F_X = X[119:673,:]
    # F_Y = Y[119:673,:]

    dxx = X[:, 119:672, :] - X[:, 120:673, :]
    dxy = Y[:, 119:672, :] - Y[:, 120:673, :]
    dyx = X[:, 119:673, 1:40] - X[:, 119:673, 0:39]
    dyy = Y[:, 119:673, 1:40] - Y[:, 119:673, 0:39]

    ds1 = torch.abs(torch.mul(dxx[:, :, :-1], dyy[:, 1:]) - torch.mul(dxy[:, :, :-1], dyx[:, 1:])) / 2
    ds2 = torch.abs(torch.mul(dxx[:, :, 1:], dyy[:, :-1]) - torch.mul(dxy[:, :, 1:], dyx[:, :-1])) / 2
    ds = ds1 + ds2

    M_T = (F_T[:, 1:, 1:] + F_T[:, 1:, :-1] + F_T[:, :-1, 1:] + F_T[:, :-1, :-1]) / 4

    Tb = torch.sum(ds * M_T, dim=(1, 2)) / torch.sum(ds, dim=(1, 2))

    return Tb


def cal_tw(X, Y, T, device=torch.device('cuda:' + str(0))):
    up_t = T[:, 119:673, 39]
    down_t = T[:, 119:673, 0]

    temp = torch.sqrt((X[:, 119:672, 39] - X[:, 120:673, 39]) ** 2 + (Y[:, 119:672, 39] - Y[:, 120:673, 39]) ** 2)
    up_dl = torch.zeros(X.shape[0], 673-119).to(device)
    up_dl[:, 1:-1] = (temp[:, 0:552] + temp[:, 1:553]) / 2
    up_dl[:, 0] = temp[:, 0] / 2
    up_dl[:, -1] = temp[:, -1] / 2

    temp = torch.sqrt((X[:, 119:672, 0] - X[:, 120:673, 0]) ** 2 + (Y[:, 119:672, 0] - Y[:, 120:673, 0]) ** 2)
    down_dl = torch.zeros(X.shape[0], 673 - 119).to(device)
    down_dl[:, 1:-1] = (temp[:, 0:552] + temp[:, 1:553]) / 2
    down_dl[:, 0] = temp[:, 0] / 2
    down_dl[:, -1] = temp[:, -1] / 2

    Tw = (torch.sum(up_t * up_dl, dim=1) + torch.sum(down_t * down_dl, dim=1)) / torch.sum(up_dl + down_dl, dim=1)

    return Tw


def cal_all(names, bound, field, grid, hflex, design, device=torch.device('cuda:' + str(0))):
    # D = float(2 * 25 * 1e-6)
    # L = float(350 * 1e-6)
    #
    # per = design[:, 3]
    # Re = design[:, 0]


    D = float(2 * 200 * 1e-6)
    L = float(3500 * 1e-6)

    per = design[:, 3]
    Re = design[:, 0]

    lamda, Cp, rho, miu = get_parameters_of_nano(per)

    X = grid[:, 0, :, :]
    Y = grid[:, 1, :, :]

    Nu = torch.zeros(field.shape[0]).to(device)
    f = torch.zeros(field.shape[0]).to(device)
    Tb = torch.zeros(field.shape[0]).to(device)

    for name in names:

        if name == "T" or "t":
            Th = bound[1, 1]
            Tc = bound[1, 0]
            T = field[:, 1, :, :] * (Th - Tc) + Tc

            Tw = cal_tw(X, Y, T, device)
            Tb = cal_tb(X, Y, T, device)
            # real_Tb = hflex[:, 2]
            # res_Tb = real_Tb-Tb
            # hflex0 = hflex[:, 1]
            # r_dT = Tw - real_Tb
            # p_dT = Tw - Tb
            # r_h = hflex0 / torch.abs(r_dT)
            # p_h = hflex0 / torch.abs(p_dT)
            # r_Nu = r_h * D / lamda
            # p_Nu = p_h * D / lamda
            h = hflex / torch.abs(Tw - Tb)

            Nu = h * D / lamda
        #
        if name == "P" or "p":
            Pi = bound[0, 1]
            Po = bound[0, 0]
            P = field[:, 0, :, :] * (Pi - Po) + Po

            vel = Re * miu / rho / D
            Dp = cal_f(X, Y, P, device)
            f = Dp * D / 2 / L / vel / vel / rho

    result = torch.stack((Nu, f, Tb, Tw), dim=1).cpu().numpy()
    # result = torch.stack((Nu, f, Tb), dim=1).cpu().numpy()

    return result

def cal_all_numpy(names, bound, field, grid, hflex, design, device=torch.device('cuda:' + str(0))):

    sample_size = field.shape[0]
    batch_size = 128
    batch_len = int(sample_size / batch_size) + 1

    Gs = torch.from_numpy(grid).to(device)
    Fs = torch.from_numpy(field).to(device)
    Ds = torch.from_numpy(design).to(device)
    Qs = torch.from_numpy(hflex).to(device)

    index = 0
    results = []

    for i in range(batch_len):

        if i == batch_len - 1:
            index = range((batch_size * i), sample_size, 1)
        else:
            index = range((batch_size * i), batch_size * (i + 1), 1)

        if i == 0:
            results = cal_all(names=names, bound=bound, field=Fs[index], grid=Gs[index],
                              hflex=Qs[index], design=Ds[index], device=device)
        else:
            result = cal_all(names=names, bound=bound, field=Fs[index], grid=Gs[index],
                             hflex=Qs[index], design=Ds[index], device=device)

            results = np.append(results, result, axis=0)

    del (Gs, Ds, Fs, Qs)
    torch.cuda.empty_cache()

    return results


if __name__ == '__main__':

    path = 'G:\\lty\\Field_reconstruct\\Re_exp_data\\exp_DG_single_10-1000.mat'
    datamat = h5py.File(path)

    fields = np.transpose(datamat['field'], (3, 2, 1, 0))[:, :, :, :].transpose([0, 3, 1, 2])
    grids = np.transpose(datamat['grids'], (3, 2, 1, 0))[:, :, :, :2].transpose([0, 3, 1, 2])
    scalar = np.transpose(datamat['scalar'], (1, 0)).squeeze()
    design = np.transpose(datamat['data'], (1, 0)).squeeze()

    NU = scalar[:, -1]
    Tbs = scalar[:, 2]
    Q = scalar[:, 1]

    bound = np.array([[0, 8e5], [292, 322], [-100, 300], [-50, 100]]).astype(np.float32)

    fields = (fields - bound[:, 0]) / (bound[:, 1] - bound[:, 0])

    import time

    s_time = time.time()

    sample_size = fields.shape[0]
    batch_size = 128
    batch_len = int(sample_size / batch_size) + 1

    device = torch.device('cuda:' + str(0))

    results = cal_all_numpy(names=("P", "T"), bound=bound, field=fields, grid=grids,
                            hflex=Q, design=design, device=device)

    e_time = time.time()
    print(e_time - s_time)

    import matplotlib.pyplot as plt
    import Utilize.data_visualization as visual

    logger = visual.matplotlib_vision("re\\")

    plt.figure(1, figsize=(20, 15))
    plt.clf()
    plt.ion()  # interactive mode on

    plt.subplot(121)
    logger.plot_regression([scalar[:, -1], np.array(results, dtype=np.float32)[:, 0]], )

    plt.subplot(122)
    plt.scatter(design[:, 0], np.array(results, dtype=np.float32)[:, 1])

    plt.pause(100)