# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    log_fileout.write('Equation name for problem: %s\n\n' % (R_dic['equa_name']))
    log_fileout.write('The type for iteration: %s\n\n' % str(R_dic['opt2Euler']))
    log_fileout.write('Network model of dealing with parameters: %s\n\n' % str(R_dic['model2NN']))

    if str.upper(R_dic['model2NN']) == 'DNN_FOURIERBASE' or str.upper(R_dic['model2NN']) == 'FOURIER_DNN':
        log_fileout.write('The input activate function: %s\n\n' % '[sin;cos]')
    else:
        log_fileout.write('The input activate function: %s\n\n' % str(R_dic['name2act_in']))

    log_fileout.write('The hidden-layer activate function: %s\n\n' % str(R_dic['name2act_hidden']))

    log_fileout.write('hidden layers for parameters: %s\n\n' % str(R_dic['hidden_layers']))

    if str.upper(R_dic['model2NN']) != 'DNN':
        log_fileout.write('The scale for frequency: %s\n\n' % str(R_dic['freq']))
        log_fileout.write('Repeat the high-frequency scale or not for NN: %s\n\n' % str(R_dic['if_repeat_High_freq']))

    log_fileout.write('The type for Loss function: %s\n\n' % str(R_dic['loss_type']))
    if str(R_dic['loss_type']) == 'lncosh_loss':
        log_fileout.write('The scale-factor for lncosh_loss: %s\n\n' % str(R_dic['lambda2lncosh']))
    log_fileout.write('The training model for all networks: %s\n\n' % str(R_dic['train_model']))

    if (R_dic['optimizer_name']).title() == 'Adam':
        log_fileout.write('optimizer:%s\n\n' % str(R_dic['optimizer_name']))
    else:
        log_fileout.write('optimizer:%s  with momentum=%f\n\n' % (R_dic['optimizer_name'], R_dic['momentum']))

    log_fileout.write('Init learning rate: %s\n\n' % str(R_dic['learning_rate']))

    if R_dic['activate_stop'] != 0:
        log_fileout.write('activate the stop_step and given_step= %s\n\n' % str(R_dic['max_epoch']))
    else:
        log_fileout.write('no activate the stop_step and given_step = default: %s\n\n' % str(R_dic['max_epoch']))

    log_fileout.write(
        'Initial penalty for difference of predict and true: %s\n\n' % str(R_dic['init_penalty2predict_true']))

    if R_dic['activate_penalty2pt_increase'] == 0:
        log_fileout.write('Unchanging penalty for predict and true!!!\n\n')
    else:
        log_fileout.write('Increasing penalty for predict and true!!!\n\n')

    log_fileout.write('The model of regular weights and biases: %s\n\n' % str(R_dic['regular_wb_model']))

    log_fileout.write('Regularizing scale for weights and biases: %s\n\n' % str(R_dic['penalty2weight_biases']))

    log_fileout.write('Size 2 training set: %s\n\n' % str(R_dic['size2train']))

    log_fileout.write('Batch-size 2 training: %s\n\n' % str(R_dic['batch_size2train']))

    log_fileout.write('Batch-size 2 testing: %s\n\n' % str(R_dic['batch_size2test']))

    if R_dic['normalize_population'] == 1:
        log_fileout.write('Do not normalize data!\n\n')
    elif (R_dic['total_population'] != R_dic['normalize_population']) and R_dic['normalize_population'] != 1:
        log_fileout.write('Utilizing scale-factor population(less than total population) to normalize data!\n\n')
        log_fileout.write('The normalization population:%s \n\n' % str(R_dic['normalize_population']))
    elif (R_dic['total_population'] == R_dic['normalize_population']) and R_dic['normalize_population'] != 1:
        log_fileout.write('Utilizing total population to normalize data!\n\n')
        log_fileout.write('The total population:%s \n\n' % str(R_dic['total_population']))
    log_fileout.flush()   # 清空缓存区


def dictionary_out2file2RK4(R_dic, log_fileout):
    log_fileout.write('Equation name for problem: %s\n\n' % (R_dic['equa_name']))
    log_fileout.write('The type for iteration: %s\n\n' % str(R_dic['opt2RK4']))
    log_fileout.write('Network model of dealing with parameters: %s\n\n' % str(R_dic['model2NN']))

    if str.upper(R_dic['model2NN']) == 'DNN_FOURIERBASE' or str.upper(R_dic['model2NN']) == 'FOURIER_DNN':
        log_fileout.write('The input activate function: %s\n\n' % '[sin;cos]')
    else:
        log_fileout.write('The input activate function: %s\n\n' % str(R_dic['name2act_in']))

    log_fileout.write('The hidden-layer activate function: %s\n\n' % str(R_dic['name2act_hidden']))

    log_fileout.write('hidden layers for parameters: %s\n\n' % str(R_dic['hidden_layers']))

    if str.upper(R_dic['model2NN']) != 'DNN':
        log_fileout.write('The scale for frequency: %s\n\n' % str(R_dic['freq']))
        log_fileout.write('Repeat the high-frequency scale or not for NN: %s\n\n' % str(R_dic['if_repeat_High_freq']))

    log_fileout.write('The type for Loss function: %s\n\n' % str(R_dic['loss_type']))
    if str(R_dic['loss_type']) == 'lncosh_loss':
        log_fileout.write('The scale-factor for lncosh_loss: %s\n\n' % str(R_dic['lambda2lncosh']))
    log_fileout.write('The training model for all networks: %s\n\n' % str(R_dic['train_model']))

    if (R_dic['optimizer_name']).title() == 'Adam':
        log_fileout.write('optimizer:%s\n\n' % str(R_dic['optimizer_name']))
    else:
        log_fileout.write('optimizer:%s  with momentum=%f\n\n' % (R_dic['optimizer_name'], R_dic['momentum']))

    log_fileout.write('Init learning rate: %s\n\n' % str(R_dic['learning_rate']))

    if R_dic['activate_stop'] != 0:
        log_fileout.write('activate the stop_step and given_step= %s\n\n' % str(R_dic['max_epoch']))
    else:
        log_fileout.write('no activate the stop_step and given_step = default: %s\n\n' % str(R_dic['max_epoch']))

    log_fileout.write(
        'Initial penalty for difference of predict and true: %s\n\n' % str(R_dic['init_penalty2predict_true']))

    if R_dic['activate_penalty2pt_increase'] == 0:
        log_fileout.write('Unchanging penalty for predict and true!!!\n\n')
    else:
        log_fileout.write('Increasing penalty for predict and true!!!\n\n')

    log_fileout.write('The model of regular weights and biases: %s\n\n' % str(R_dic['regular_wb_model']))

    log_fileout.write('Regularizing scale for weights and biases: %s\n\n' % str(R_dic['penalty2weight_biases']))

    log_fileout.write('Size 2 training set: %s\n\n' % str(R_dic['size2train']))

    log_fileout.write('Batch-size 2 training: %s\n\n' % str(R_dic['batch_size2train']))

    log_fileout.write('Batch-size 2 testing: %s\n\n' % str(R_dic['batch_size2test']))

    if R_dic['normalize_population'] == 1:
        log_fileout.write('Do not normalize data!\n\n')
    elif (R_dic['total_population'] != R_dic['normalize_population']) and R_dic['normalize_population'] != 1:
        log_fileout.write('Utilizing scale-factor population(less than total population) to normalize data!\n\n')
        log_fileout.write('The normalization population:%s \n\n' % str(R_dic['normalize_population']))
    elif (R_dic['total_population'] == R_dic['normalize_population']) and R_dic['normalize_population'] != 1:
        log_fileout.write('Utilizing total population to normalize data!\n\n')
        log_fileout.write('The total population:%s \n\n' % str(R_dic['total_population']))
    log_fileout.flush()   # 清空缓存区

# 记录字典中的一些设置
def dictionary_out2file2GD2INN(R_dic, log_fileout):
    log_fileout.write('ODE type:%s\n\n' % R_dic['ODE_type'])
    log_fileout.write('Equation name: %s\n\n' % (R_dic['equa_name']))
    log_fileout.write('Network model of dealing with equation: %s\n\n' % str(R_dic['model2Eq']))

    if str.upper(R_dic['model2Eq']) == 'DNN_FOURIERBASE' or str.upper(R_dic['model2Eq']) == 'FOURIER_DNN':
        log_fileout.write('The input activate function for EqNN: %s\n\n' % '[sin;cos]')
    else:
        log_fileout.write('The input activate function for EqNN: %s\n\n' % str(R_dic['name_actIn2Eq']))

    log_fileout.write('The hidden-layer activate function for EqNN: %s\n\n' % str(R_dic['name_actHidden2Eq']))

    log_fileout.write('The output activate function for EqNN: %s\n\n' % str(R_dic['name_actOut2Eq']))

    log_fileout.write('hidden layers for EqNN: %s\n\n' % str(R_dic['hidden2Eq']))

    if str.upper(R_dic['model2Eq']) != 'DNN':
        log_fileout.write('The scale for EqNN: %s\n\n' % str(R_dic['freq2Eq']))
        log_fileout.write('Repeat the high-frequency scale or not for EqNN: %s\n\n' % str(R_dic['repeat_High_freq2Eq']))

    log_fileout.write('Network model of dealing with parameter: %s\n\n' % str(R_dic['model2Para']))

    if str.upper(R_dic['model2Para']) == 'DNN_FOURIERBASE' or str.upper(R_dic['model2Para']) == 'FOURIER_DNN':
        log_fileout.write('The input activate function for ParaNN: %s\n\n' % '[sin;cos]')
    else:
        log_fileout.write('The input activate function for ParaNN: %s\n\n' % str(R_dic['name_actIn2Para']))

    log_fileout.write('The hidden-layer activate function for ParaNN: %s\n\n' % str(R_dic['name_actHidden2Para']))

    log_fileout.write('The output activate function for ParaNN: %s\n\n' % str(R_dic['name_actOut2Para']))

    log_fileout.write('hidden layers for ParaNN: %s\n\n' % str(R_dic['hidden2Para']))

    if str.upper(R_dic['model2Para']) != 'DNN':
        log_fileout.write('The scale for ParaNN: %s\n\n' % str(R_dic['freq2Para']))
        log_fileout.write(
            'Repeat the high-frequency scale or not for ParaNN: %s\n\n' % str(R_dic['repeat_High_freq2Para']))

    log_fileout.write('The type for Loss function: %s\n\n' % str(R_dic['loss_type']))
    if str(R_dic['loss_type']) == 'lncosh_loss':
        log_fileout.write('The scale-factor for lncosh_loss: %s\n\n' % str(R_dic['lambda2lncosh']))
    log_fileout.write('The training model for all networks: %s\n\n' % str(R_dic['train_model']))

    if (R_dic['optimizer_name']).title() == 'Adam':
        log_fileout.write('optimizer:%s\n\n' % str(R_dic['optimizer_name']))
    else:
        log_fileout.write('optimizer:%s  with momentum=%f\n\n' % (R_dic['optimizer_name'], R_dic['momentum']))

    log_fileout.write('Init learning rate: %s\n\n' % str(R_dic['learning_rate']))

    if R_dic['activate_stop'] != 0:
        log_fileout.write('activate the stop_step and given_step= %s\n\n' % str(R_dic['max_epoch']))
    else:
        log_fileout.write('no activate the stop_step and given_step = default: %s\n\n' % str(R_dic['max_epoch']))

    log_fileout.write(
        'Initial penalty for difference of predict and true: %s\n\n' % str(R_dic['init_penalty2predict_true']))

    if R_dic['activate_penalty2pt_increase'] == 0:
        log_fileout.write('Unchanging penalty for predict and true!!!\n\n')
    else:
        log_fileout.write('Increasing penalty for predict and true!!!\n\n')

    log_fileout.write('The model of regular weights and biases: %s\n\n' % str(R_dic['regular_wb_model']))

    log_fileout.write('Regularizing scale for weights and biases: %s\n\n' % str(R_dic['penalty2weight_biases']))

    log_fileout.write('Size 2 training set: %s\n\n' % str(R_dic['size2train']))

    log_fileout.write('Batch-size 2 training: %s\n\n' % str(R_dic['batch_size2train']))

    log_fileout.write('Batch-size 2 testing: %s\n\n' % str(R_dic['batch_size2test']))

    if R_dic['normalize_population'] == 1:
        log_fileout.write('Do not normalize data!\n\n')
    elif (R_dic['total_population'] != R_dic['normalize_population']) and R_dic['normalize_population'] != 1:
        log_fileout.write('Utilizing scale-factor population(less than total population) to normalize data!\n\n')
        log_fileout.write('The normalization population:%s \n\n' % str(R_dic['normalize_population']))
    elif (R_dic['total_population'] == R_dic['normalize_population']) and R_dic['normalize_population'] != 1:
        log_fileout.write('Utilizing total population to normalize data!\n\n')
        log_fileout.write('The total population:%s \n\n' % str(R_dic['total_population']))
    log_fileout.flush()   # 清空缓存区


def print_training2ThreeNet(epoch, run_time, tmp_lr, penalty_wb2beta, penalty_wb2gamma, penalty_wb2mu, loss_s, loss_i,
                        loss_r, loss_d, loss_all, log_out=None):
    print('train epoch: %d, time: %.4f' % (epoch, run_time))
    print('learning rate: %.12f' % tmp_lr)
    print('penalty weights and biases for Beta: %.10f' % penalty_wb2beta)
    print('penalty weights and biases for Gamma: %.10f' % penalty_wb2gamma)
    print('penalty weights and biases for Mu: %.10f' % penalty_wb2mu)
    print('loss for S: %.10f' % loss_s)
    print('loss for I: %.10f' % loss_i)
    print('loss for R: %.10f' % loss_r)
    print('loss for D: %.10f' % loss_d)
    print('total loss: %.10f\n' % loss_all)

    log_out.write('train epoch: %d,time: %.4f \n' % (epoch, run_time))
    log_out.write('learning rate: %.12f \n' % tmp_lr)
    log_out.write('penalty weights and biases for Beta: %.10f \n' % penalty_wb2beta)
    log_out.write('penalty weights and biases for Gamma: %.10f \n' % penalty_wb2gamma)
    log_out.write('penalty weights and biases for Mu: %.10f \n' % penalty_wb2mu)
    log_out.write('loss for S: %.10f \n' % loss_s)
    log_out.write('loss for I: %.10f \n' % loss_i)
    log_out.write('loss for R: %.10f \n' % loss_r)
    log_out.write('loss for D: %.10f \n' % loss_d)
    log_out.write('total loss: %.10f \n\n' % loss_all)
    log_out.flush()   # 清空缓存区


def print_training2OneNet(epoch, run_time, tmp_lr, penalty_wb2Paras, loss_s, loss_i,
                          loss_r, loss_d, loss_all, log_out=None):
    print('train epoch: %d, time: %.4f' % (epoch, run_time))
    print('learning rate: %.10f' % tmp_lr)
    print('penalty weights and biases for Paras: %.10f' % penalty_wb2Paras)
    print('loss for S to train: %.10f' % loss_s)
    print('loss for I to train: %.10f' % loss_i)
    print('loss for R to train: %.10f' % loss_r)
    print('loss for D to train: %.10f' % loss_d)
    print('total loss to train: %.10f\n' % loss_all)

    log_out.write('train epoch: %d,time: %.4f \n' % (epoch, run_time))
    log_out.write('learning rate: %.12f \n' % tmp_lr)
    log_out.write('penalty weights and biases for Paras: %.8f \n' % penalty_wb2Paras)
    log_out.write('loss for S to train: %.8f \n' % loss_s)
    log_out.write('loss for I to train: %.8f \n' % loss_i)
    log_out.write('loss for R to train: %.8f \n' % loss_r)
    log_out.write('loss for D to train: %.8f \n' % loss_d)
    log_out.write('total loss to train: %.8f \n\n' % loss_all)
    log_out.flush()   # 清空缓存区


def print_BFGStraining2OneNet(epoch, run_time, tmp_lr, loss_all, log_out=None):
    print('train epoch: %d, time: %.4f' % (epoch, run_time))
    print('learning rate: %.10f' % tmp_lr)
    print('total loss to train: %.10f\n' % loss_all)

    log_out.write('train epoch: %d,time: %.4f \n' % (epoch, run_time))
    log_out.write('learning rate: %.12f \n' % tmp_lr)
    log_out.write('total loss to train: %.8f \n\n' % loss_all)
    log_out.flush()   # 清空缓存区


def print_test2OneNet(mse2s,  mse2i, mse2r, mse2d, rel2s, rel2i, rel2r, rel2d, log_out=None):
    print('MSE of S to testing: %.8f' % mse2s)
    print('MSE of I to testing: %.8f' % mse2i)
    print('MSE of R to testing: %.8f' % mse2r)
    print('MSE of D to testing: %.8f\n' % mse2d)

    print('REL of S to testing: %.8f' % rel2s)
    print('REL of I to testing: %.8f' % rel2i)
    print('REL of R to testing: %.8f' % rel2r)
    print('REL of D to testing: %.8f\n' % rel2d)

    log_out.write('MSE of S to testing: %.8f \n' % mse2s)
    log_out.write('MSE of I to testing: %.8f \n' % mse2i)
    log_out.write('MSE of R to testing: %.8f \n' % mse2r)
    log_out.write('MSE of D to testing: %.8f \n\n' % mse2d)

    log_out.write('REL of S to testing: %.8f \n' % rel2s)
    log_out.write('REL of I to testing: %.8f \n' % rel2i)
    log_out.write('REL of R to testing: %.8f \n' % rel2r)
    log_out.write('REL of D to testing: %.8f \n\n' % rel2d)

    log_out.flush()   # 清空缓存区


def print_testFix_paras2OneNet(mse2s,  mse2i, mse2r, mse2d, rel2s, rel2i, rel2r, rel2d, log_out=None):
    print('MSE for S with fixed parameters: %.8f' % mse2s)
    print('MSE for I with fixed parameters: %.8f' % mse2i)
    print('MSE for R with fixed parameters: %.8f' % mse2r)
    print('MSE for D with fixed parameters: %.8f\n' % mse2d)

    print('REL for S with fixed parameters: %.8f' % rel2s)
    print('REL for I with fixed parameters: %.8f' % rel2i)
    print('REL for R with fixed parameters: %.8f' % rel2r)
    print('REL for D with fixed parameters: %.8f\n' % rel2d)

    log_out.write('MSE for S with fixed parameters: %.8f \n' % mse2s)
    log_out.write('MSE for I with fixed parameters: %.8f \n' % mse2i)
    log_out.write('MSE for R with fixed parameters: %.8f \n' % mse2r)
    log_out.write('MSE for D with fixed parameters: %.8f \n\n' % mse2d)

    log_out.write('REL for S with fixed parameters: %.8f \n' % rel2s)
    log_out.write('REL for I with fixed parameters: %.8f \n' % rel2i)
    log_out.write('REL for R with fixed parameters: %.8f \n' % rel2r)
    log_out.write('REL for D with fixed parameters: %.8f \n\n' % rel2d)

    log_out.flush()   # 清空缓存区


def print_training2TwoNet_GD2INN(epoch, run_time, tmp_lr, penalty_wb, loss_sdata, loss_sderi, loss_s,
                                 loss_idata, loss_ideri, loss_i, loss_rdata, loss_rderi, loss_r, loss_ddata,
                                 loss_dderi, loss_d, loss_all, log_out=None):
    print('train epoch: %d, time: %.4f' % (epoch, run_time))
    print('learning rate: %.10f' % tmp_lr)
    print('penalty weights and biases for ParaNN and EqNN: %.10f' % penalty_wb)

    print('loss of S for observed and predicted to train: %.10f' % loss_sdata)
    print('loss of I for observed and predicted to train: %.10f' % loss_idata)
    print('loss of R for observed and predicted to train: %.10f' % loss_rdata)
    print('loss of D for observed and predicted to train: %.10f' % loss_ddata)

    print('loss of S for auto-grad and estimated-grad to train: %.10f' % loss_sderi)
    print('loss of I for auto-grad and estimated-grad to train: %.10f' % loss_ideri)
    print('loss of R for auto-grad and estimated-grad to train: %.10f' % loss_rderi)
    print('loss of D for auto-grad and estimated-grad to train: %.10f' % loss_dderi)

    print('loss for S to train: %.10f' % loss_s)
    print('loss for I to train: %.10f' % loss_i)
    print('loss for R to train: %.10f' % loss_r)
    print('loss for D to train: %.10f' % loss_d)
    print('total loss to train: %.10f\n' % loss_all)

    log_out.write('train epoch: %d,time: %.4f \n' % (epoch, run_time))
    log_out.write('learning rate: %.12f \n' % tmp_lr)
    log_out.write('penalty weights and biases for Paras: %.8f \n' % penalty_wb)

    log_out.write('loss of S for observed and predicted to train: %.8f \n' % loss_sdata)
    log_out.write('loss of S for auto-grad and estimated-grad to train: %.8f \n' % loss_sderi)
    log_out.write('loss for S to train: %.8f \n' % loss_s)

    log_out.write('loss of I for observed and predicted to train: %.8f \n' % loss_idata)
    log_out.write('loss of I for auto-grad and estimated-grad to train: %.8f \n' % loss_ideri)
    log_out.write('loss for I to train: %.8f \n' % loss_i)

    log_out.write('loss of R for observed and predicted to train: %.8f \n' % loss_rdata)
    log_out.write('loss of R for auto-grad and estimated-grad to train: %.8f \n' % loss_rderi)
    log_out.write('loss for R to train: %.8f \n' % loss_r)

    log_out.write('loss of D for observed and predicted to train: %.8f \n' % loss_ddata)
    log_out.write('loss of D for auto-grad and estimated-grad to train: %.8f \n' % loss_dderi)
    log_out.write('loss for D to train: %.8f \n' % loss_d)
    log_out.write('total loss to train: %.8f \n\n' % loss_all)
    log_out.flush()   # 清空缓存区
