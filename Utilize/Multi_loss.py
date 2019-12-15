import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

class Conver_loss(nn.Module):
    def __init__(self):
        super().__init__()

    # 计算纳米流体热物理性质
    def get_parameters_of_nano(self, per):
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

    def one_derivate(self, X, Y, Varis):
        dV_x = Varis[:, 2:, 1:-1] - Varis[:, 0:-2, 1:-1]
        dV_y = Varis[:, 1:-1, 2:] - Varis[:, 1:-1, 0:-2]
        dx = (X[:, 2:, 1:-1] - Y[:, 0:-2, 1:-1]) / 2
        dy = (Y[:, 1:-1, 2:] - Y[:, 1:-1, 0:-2]) / 2
        dV_dx = dV_x / dx
        dV_dy = dV_y / dy
        return dV_dx, dV_dy

    def two_derivate(self, X, Y, Varis):
        dV_x = Varis[:, 1:, 1:-1] - Varis[:, 0:-1, 1:-1]
        dV_y = Varis[:, 1:-1, 1:] - Varis[:, 1:-1, 0:-1]
        dx = (X[:, 2:, 1:-1] - Y[:, 0:-2, 1:-1]) / 2
        dy = (Y[:, 1:-1, 2:] - Y[:, 1:-1, 0:-2]) / 2
        ddV_x = dV_x[:, 1:, :] - dV_x[:, 0:-1, :]
        ddV_y = dV_y[:, :, 1:] - dV_y[:, :, 0:-1]
        dV_dx2 = ddV_x / dx ** 2
        dV_dy2 = ddV_y / dy ** 2
        return dV_dx2, dV_dy2

    def cal_cont(self, X, Y, U, V, device=torch.device('cuda:' + str(0))):
        dU_dx, _ = self.one_derivate(X, Y, U)
        _, dV_dy = self.one_derivate(X, Y, V)
        res_c = dU_dx + dV_dy
        return res_c

    def expand_torch(self, V, shape):
        a = V.expand(shape[2] - 2, shape[1] - 2, shape[0])
        a = a.permute(2, 1, 0)
        return a

    def cal_mass(self, X, Y, P, U, V, design, device=torch.device('cuda:' + str(0))):
        dU_dx, dU_dy = self.one_derivate(X, Y, U)
        dV_dx, dV_dy = self.one_derivate(X, Y, V)
        dP_dx, dP_dy = self.one_derivate(X, Y, P)
        dU_dx2, dU_dy2 = self.two_derivate(X, Y, U)
        dV_dx2, dV_dy2 = self.two_derivate(X, Y, V)

        per = design[:, 3]
        phy = list(self.get_parameters_of_nano(per))
        shape = X.size()
        lamda, Cp, rho, miu = [self.expand_torch(x, shape) for x in phy]

        res_u = U[:, 1:-1, 1:-1] * dU_dx
        res_u = res_u + V[:, 1:-1, 1:-1] * dU_dy
        res_u = res_u + 1 / rho * dP_dx
        res_u = res_u - miu * (dU_dx2 + dU_dy2)
        res_v = U[:, 1:-1, 1:-1] * dV_dx + V[:, 1:-1, 1:-1] * dV_dy + 1 / rho * dP_dy - miu * (dV_dx2 + dV_dy2)
        return res_u, res_v

    def cal_ener(self, X, Y, P, T, U, V, design, device=torch.device('cuda:' + str(0))):
        dT_dx, dT_dy = self.one_derivate(X, Y, T)
        dT_dx2, dT_dy2 = self.two_derivate(X, Y, T)

        per = design[:, 3]
        phy = list(self.get_parameters_of_nano(per))
        shape = X.size()
        lamda, Cp, rho, miu = [self.expand_torch(x, shape) for x in phy]

        res_T = U[:, 1:-1, 1:-1] * dT_dx + V[:, 1:-1, 1:-1] * dT_dy + lamda / (rho * Cp) * (dT_dx2 + dT_dy2)
        return res_T

    def cal_all(self, field, grid, design, device=torch.device('cuda:' + str(0))):

        X = grid[:, 0, :, :]
        Y = grid[:, 1, :, :]
        P = field[:, 0, :, :]
        T = field[:, 1, :, :]
        U = field[:, 2, :, :]
        V = field[:, 3, :, :]
        res_c = self.cal_cont(X, Y, U, V)
        res_u, res_v = self.cal_mass(X, Y, P, U, V, design)
        res_T = self.cal_ener(X, Y, P, T, U, V, design)
        result = [torch.mean(res_c), torch.mean(res_u), torch.mean(res_v), torch.mean(res_T)]

        return result.cpu().numpy()

    def forward(self, field, grid, design, device=torch.device('cuda:' + str(0))):

        sample_size = field.shape[0]
        batch_size = 128
        batch_len = int(sample_size / batch_size) + 1

        Gs = grid.to(device)
        Fs = field.to(device)
        Ds = design.to(device)

        index = 0
        results = []

        for i in range(batch_len):

            if i == batch_len - 1:
                index = range((batch_size * i), sample_size, 1)
            else:
                index = range((batch_size * i), batch_size * (i + 1), 1)

            if i == 0:
                results = self.cal_all(field=Fs[index], grid=Gs[index],
                                  design=Ds[index], device=device)
            else:
                result = self.cal_all(field=Fs[index], grid=Gs[index],
                                 design=Ds[index], device=device)

                results = np.append(results, result, axis=0)

        del (Gs, Ds, Fs)
        torch.cuda.empty_cache()
        a = 2
        return results

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


if __name__ == '__main__':

   field = torch.randn(10, 4, 792, 40)
   grid = torch.randn(10, 2, 792, 40)
   design = torch.randn(10, 10) * 0.1
   C_loss = Conver_loss()
   a = C_loss(field, grid, design)
