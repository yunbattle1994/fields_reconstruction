import numpy as np
import torch
import h5py



def get_parameters_of_foil():
    # 参考值
    Density = 1.225  # kg/m3
    U_inf = 1  # m/s
    L = 1  # m

    return Density, U_inf, L


def caculate(field_name, fields_true, fields_pred, grids, bound):
    # 提取信息
    (t_num, cir_num, rad_num, para_num) = np.shape(fields_true)

    results_true = []
    results_pred = []

    for (f_t, f_p, grid) in zip(fields_true, fields_pred, grids):
        result_t = cal_force(f_t, grid)
        results_true.append(result_t)

        result_p = cal_force(f_p, grid)
        results_pred.append(result_p)

    return results_true, results_pred


def cal_force(fields, grids):
    Density, U_inf, L = get_parameters_of_foil()

    Ref_F = 0.5 * Density * U_inf * U_inf * L

    pt = fields[:, -1, 0]
    # ut = fields[:, -1, 1]
    # vt = fields[:, -1, 2]

    (cir_num,) = np.shape(pt)
    ind1 = [i for i in range(cir_num)]
    ind2 = [i for i in range(1, cir_num)]
    ind2.append(0)

    pt_ave = (pt[ind1] + pt[ind2]) / 2.
    coordt_n_middle = 0.5 * (grids[ind1, -1, :] + grids[ind2, -1, :])
    T_vector = grids[ind2, -1, :] - grids[ind1, -1, :]
    N_vector = np.dot(T_vector, np.array([[0, -1], [1, 0]]))

    T_norm = np.linalg.norm(T_vector, axis=1)
    N_norm = np.linalg.norm(N_vector, axis=1)

    Ft_n = pt_ave * T_norm
    Fx = Ft_n * N_vector[:, 0] / N_norm
    Fy = Ft_n * N_vector[:, 1] / N_norm

    Mz = -Fx * coordt_n_middle[:, 1] + Fy * coordt_n_middle[:, 0]

    Fx = np.sum(Fx)
    Fy = np.sum(Fy)
    Mz = np.sum(Mz)
    Cd = Fx / Ref_F
    Cl = Fy / Ref_F

    return Fx, Fy, Mz, Cd, Cl, Ref_F

