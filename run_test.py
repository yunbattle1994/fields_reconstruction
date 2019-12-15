import os
import torch
import numpy as np
import net_model_DeRes as model
import tur_DG_data as loader


if __name__ == '__main__':

    Run_Mode = 'train'
    print("Mode:   " + Run_Mode)

    # 计算设备
    CUDA_ID = 4
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_ID)
    device_name = torch.device('cuda:' + str(0))
    # device_name = torch.device('cpu:' + str(0))
    print("use GPU_id {:d}".format(CUDA_ID))


    # 网络参数设置
    model_name = 'DeRes18'
    loss_weight = [1, 0.1]
    learning_rate = 0.001
    boundary_epochs = [100, 150, ]
    grad_rates = [1, 0.5, 0.1]
    decay_rates = [1, 0.1, 0.01]
    start_epoch = 0
    end_epoch = 201
    display_batch = 300
    weight_decay = 1e-5
    hidden_size = 256

    # 数据格式设置
    std = 0
    train_size = 0.2
    limit_size = 0.5

    field_name = ("P", "T", "U", "V")
    # field_name = ( "T", "U", "V")
    name_index = {"P": 0, "T": 1, "U": 2, "V": 3, "p": 0, "t": 1, "u": 2, "v": 3}
    field_index = [name_index[name] for name in field_name]
    # field_bound = np.array([[-1, 8e5], [292, 350], [-4, 40], [-4, 4]]).astype(np.float32)  # Re_100-1000
    field_bound = np.array([[-3e4, 5e4], [292, 325], [-2, 10], [-3, 1]]).astype(np.float32)  # dim_pro

    # points = np.array([0, 1, 1, 1, 1, 1, 0])
    # points = np.array([1, 1, 2, 1, 2, 1, 1])
    # points = np.array([2, 2, 4, 2, 4, 2, 2])
    points = np.array([2, 4, 8, 4, 8, 4, 2])
    layout = 'B'

    # 文件位置
    data_name = "dim_pro8"
    input_dir = data_name + '_data'
    test_dir = data_name + '_test'
    result_dir = data_name + "_result"
    # fold_name = '\\point_sd'
    fold_name = ''

    if_design = True
    if_grid = True
    # save_name = "points_" + str(np.size(points) * 2) + "_train_size_" + str(train_size) \
    save_name = "_train_size_" + str(train_size) + "@" + str(limit_size) \
                + "_loss_weight" + str(loss_weight) + '_sd' + str(std) \
                + "_lr_" + str(learning_rate) + "_all_eval_grad_decay" + str(grad_rates)

    if if_design:
        save_name = result_dir + "\\designs_" + str(5) + save_name + ''
    else:
        save_name = result_dir + fold_name + "\\points_" + layout + '_' + str(np.sum(points)+2) + save_name+''


    if not os.path.exists(save_name):
        os.makedirs(save_name)

    print("input_dir: {:s},  test_dir: {:s}, result_dir: {:s}".
          format(input_dir, test_dir, result_dir))

    print("save name: {:s},  model name: {:s}, field name: {:s}".
          format(save_name, model_name, str(field_name)))

    input_size = (10, len(field_name), 792, 40)
    hidden_size = 10

    # 网络建立
    net_model = model.model(model_name=model_name, input_size=input_size, hidden_size=hidden_size)
    net_model.net_set(save_name, model_name=model_name, device_name=device_name, field_name=field_name,
                      field_bound=field_bound,
                      field_index=field_index, loss_weight=loss_weight, is_resume=False)

    print("data:name: {:s},  model name: {:s}, save path: {:s}, use GPU_id {:d}".
          format(save_name, model_name, net_model.model_path, CUDA_ID))

    # 数据载入
    # test_loader = loader.get_test_loaders(test_dir=test_dir, field_bound=field_bound, batch_size=12, points=points, if_design=if_design)
    # train_loader, eval_loader = loader.get_GD_loaders(data_aug=True, batch_size=16, eval_batch_size=64, num_workers=0)

    # 数据载入
    test_loader, _ = loader.get_DG_loaders(std=0, train_size=0.5, input_dir=input_dir,
                                           field_name=field_name, field_index=field_index, field_bound=field_bound,
                                           points=points, batch_size_train=1, batch_size_eval=1, if_design=if_design,
                                           matname="try")

    net_model.load_model(resume_pattern=None, model_name="lowest_loss_model.pth")

    net_model.predict(test_loader, epoch='f')

    del net_model