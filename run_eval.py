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




    # 数据载入
    train_loader, eval_loader = loader.get_DG_loaders(input_dir=input_dir,
                                                      std=std, train_size=train_size, batch_size_train=2,
                                                      batch_size_eval=2, limit_size=limit_size,
                                                      field_name=field_name, field_index=field_index,
                                                      field_bound=field_bound,
                                                      points=points, layout=layout, if_design=if_design,
                                                      matname="all")
    # train_loader, eval_loader = loader.get_DG_test_loaders(std=std, train_size=0.5, input_dir=input_dir,
    #                                                   field_name=field_name, field_index=field_index,
    #                                                   field_bound=field_bound,
    #                                                   points=points, batch_size_train=32, batch_size_eval=64,
    #                                                   if_design=if_design, test_size=0.5)

    ind_eval = eval_loader.dataset.indexs
    np.savetxt(save_name + '\\ind_eval.txt', ind_eval, fmt='%d')

    ind_train = train_loader.dataset.indexs
    np.savetxt(save_name + '\\ind_train.txt', ind_train, fmt='%d')

    # input_size [channel, ...]
    input_size = (10, len(field_name), 792, 40)
    hidden_size = train_loader.dataset.inputs.shape[1]  # np.size(points) * 2

    # 网络建立
    net_model = model.model(model_name=model_name, input_size=input_size, hidden_size=hidden_size)
    net_model.net_set(save_name, model_name=model_name, device_name=device_name, field_name=field_name,
                      field_bound=field_bound, field_index=field_index, loss_weight=loss_weight, is_resume=False)

    model_name = "epoch_400_model"
    model_path = os.path.join(net_model.save_path, model_name)
    if not os.path.exists(model_path):
        print('path')
        os.makedirs(model_path)
    # net_model.load_model(resume_pattern=None, model_name="lowest_loss_model.pth")
    net_model.load_model(resume_pattern="direct", model_name=model_name + ".pth")

    # 统计场误差
    t_field_error = np.array(net_model.process_info["train_field_errors"], dtype=np.float32)
    e_field_error = np.array(net_model.process_info["eval_field_errors"], dtype=np.float32)
    # 统计场得分
    t_field_score = np.array(net_model.process_info["train_result_scores"], dtype=np.float32)
    e_field_score = np.array(net_model.process_info["eval_result_scores"], dtype=np.float32)
    # 统计场得分
    t_field_flowres = np.array(net_model.process_info['train_flow_results'], dtype=np.float32)
    e_field_flowres = np.array(net_model.process_info['eval_flow_results'], dtype=np.float32)

    # 统计三种损失
    t_loss_T = np.array(net_model.process_info["train_loss_logs"], dtype=np.float32)
    e_loss_T = np.array(net_model.process_info["eval_loss_logs"], dtype=np.float32)
    # t_loss_G = np.array(net_model.process_info["eval_G_loss"], dtype=np.float32)
    # e_loss_G = np.array(net_model.process_info["train_G_loss"], dtype=np.float32)
    # t_loss_F = np.array(net_model.process_info["eval_F_loss"], dtype=np.float32)
    # e_loss_F = np.array(net_model.process_info["train_F_loss"], dtype=np.float32)

    import scipy.io as sio

    sio.savemat(net_model.save_path + "\\" + model_name + "\\process.mat",
                {'t_loss_T': t_loss_T, 'e_loss_T': e_loss_T,
                 # 't_loss_G': t_loss_G, 'e_loss_G': e_loss_G,
                 # 't_loss_F': t_loss_F, 'e_loss_F': e_loss_F,
                 't_field_S': t_field_score, 'e_field_S': e_field_score,
                 't_field_E': t_field_error, 'e_field_E': e_field_error,
                 't_field_R': t_field_flowres, 'e_field_R': e_field_flowres})

    loss_t, error_t, flowres_t, name_t = net_model.evaluation(train_loader, mode='eval')
    loss_e, error_e, flowres_e, name_e = net_model.evaluation(eval_loader, mode='eval')
    # _, _, _, _, error_e, flowres_e, design_e = net_model.evaluation(eval_loader)

    flowres_true = np.concatenate((flowres_t.true, flowres_e.true), axis=0)
    flowres_pred = np.concatenate((flowres_t.pred, flowres_e.pred), axis=0)

    flowres_true = np.concatenate((flowres_t.true, flowres_e.true), axis=0)
    flowres_pred = np.concatenate((flowres_t.pred, flowres_e.pred), axis=0)
    indexs = np.concatenate((name_t.true, name_e.true), axis=0)

    sio.savemat(net_model.save_path + "\\" + model_name + "\\evaluate.mat", {'t_loss': loss_t, 'e_loss': loss_e,
                                                                             't_field': error_t,
                                                                             'e_field': error_e,
                                                                             'index': indexs,
                                                                             'flowres_real': flowres_true,
                                                                             'flowres_pred': flowres_pred,
                                                                             'R2': flowres_e.r2_score()
                                                                             })

    del net_model