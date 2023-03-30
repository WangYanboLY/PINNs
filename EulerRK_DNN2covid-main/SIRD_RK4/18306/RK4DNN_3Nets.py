"""
@author: LXA
 Date: 2022 年 9 月 10 日
"""
import os
import sys
import torch
import torch.nn as tn
import numpy as np
import matplotlib
import platform
import shutil
import time

import DNN_base
import saveData
import plotData
import logUtils
import dataUtils
import RK4_SIRD
import itertools


class RK4DNN(tn.Module):
    def __init__(self, input_dim=1, out_dim=3, hidden_layer=None, Model_name='DNN', name2actIn='relu',
                 name2actHidden='relu', name2actOut='linear', opt2regular_WB='L2', type2numeric='float32',
                 factor2freq=None, sFourier=1.0, repeat_highFreq=True, use_gpu=False, No2GPU=0):
        super(RK4DNN, self).__init__()
        if 'DNN' == str.upper(Model_name):
            self.BetaNN = DNN_base.Pure_DenseNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, scope2W='Ws2Beta',
                scope2B='Bs2Beta', type2float=type2numeric)
            self.GammaNN = DNN_base.Pure_DenseNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, scope2W='Ws2Gamma',
                scope2B='Bs2Gamma', type2float=type2numeric)
            self.MuNN = DNN_base.Pure_DenseNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, scope2W='Ws2Mu',
                scope2B='Bs2Mu', type2float=type2numeric)
        elif 'SCALE_DNN' == str.upper(Model_name) or 'DNN_SCALE' == str.upper(Model_name):
            self.BetaNN = DNN_base.Dense_ScaleNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, scope2W='Ws2Beta',
                scope2B='Bs2Beta', repeat_Highfreq=repeat_highFreq, type2float=type2numeric, to_gpu=use_gpu,
                gpu_no=No2GPU)
            self.GammaNN = DNN_base.Dense_ScaleNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, scope2W='Ws2Gamma',
                scope2B='Bs2Gamma', repeat_Highfreq=repeat_highFreq, type2float=type2numeric, to_gpu=use_gpu,
                gpu_no=No2GPU)
            self.MuNN = DNN_base.Dense_ScaleNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, scope2W='Ws2Mu',
                scope2B='Bs2Mu', repeat_Highfreq=repeat_highFreq, type2float=type2numeric, to_gpu=use_gpu,
                gpu_no=No2GPU)
        elif 'FOURIER_DNN' == str.upper(Model_name) or 'DNN_FOURIERBASE' == str.upper(Model_name):
            self.BetaNN = DNN_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, scope2W='Ws2Beta',
                scope2B='Bs2Beta', repeat_Highfreq=repeat_highFreq, type2float=type2numeric, to_gpu=use_gpu,
                gpu_no=No2GPU)
            self.GammaNN = DNN_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, scope2W='Ws2Gamma',
                scope2B='Bs2Gamma', repeat_Highfreq=repeat_highFreq, type2float=type2numeric, to_gpu=use_gpu,
                gpu_no=No2GPU)
            self.MuNN = DNN_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, scope2W='Ws2Mu',
                scope2B='Bs2Mu', repeat_Highfreq=repeat_highFreq, type2float=type2numeric, to_gpu=use_gpu,
                gpu_no=No2GPU)
        elif 'FOURIER_SUBDNN' == str.upper(Model_name) or 'SUBDNN_FOURIERBASE' == str.upper(Model_name):
            self.BetaNN = DNN_base.Fourier_SubNets3D(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, scope2W='Ws2Beta',
                scope2B='Bs2Beta', repeat_Highfreq=repeat_highFreq, type2float=type2numeric, to_gpu=use_gpu,
                gpu_no=No2GPU, num2subnets=len(factor2freq))
            self.GammaNN = DNN_base.Fourier_SubNets3D(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, scope2W='Ws2Gamma',
                scope2B='Bs2Gamma', repeat_Highfreq=repeat_highFreq, type2float=type2numeric, to_gpu=use_gpu,
                gpu_no=No2GPU, num2subnets=len(factor2freq))
            self.MuNN = DNN_base.Fourier_SubNets3D(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, scope2W='Ws2Mu',
                scope2B='Bs2Mu', repeat_Highfreq=repeat_highFreq, type2float=type2numeric, to_gpu=use_gpu,
                gpu_no=No2GPU, num2subnets=len(factor2freq))

        if type2numeric == 'float32':
            self.float_type = torch.float32
        elif type2numeric == 'float64':
            self.float_type = torch.float64
        elif type2numeric == 'float16':
            self.float_type = torch.float16

        self.use_gpu = False
        if use_gpu:
            self.opt2device = 'cuda:' + str(No2GPU)
        else:
            self.opt2device = 'cpu'

        self.input_dim = input_dim
        self.factor2freq = factor2freq
        self.sFourier = sFourier
        self.opt2regular_WB = opt2regular_WB

    def RK4Iteration(self, t=None, input_size=100, s_obs=None, i_obs=None, r_obs=None, d_obs=None,
                     loss_type='ritz_loss', scale2lncosh=0.1, scale2beta=1.0, scale2gamma=0.1, scale2mu=0.05):
        assert (t is not None)
        assert (s_obs is not None)
        assert (i_obs is not None)
        assert (r_obs is not None)
        assert (d_obs is not None)

        shape2t = t.shape
        lenght2t_shape = len(shape2t)
        assert (lenght2t_shape == 2)
        assert (shape2t[-1] == 1)

        nn_beta = self.BetaNN(t, scale=self.factor2freq, sFourier=self.sFourier)
        nn_gamma = self.GammaNN(t, scale=self.factor2freq, sFourier=self.sFourier)
        nn_mu = self.MuNN(t, scale=self.factor2freq, sFourier=self.sFourier)
        ParamsNN = torch.concat([nn_beta, nn_gamma, nn_mu], dim=-1)

        lsit2S_Pre, lsit2I_Pre, lsit2R_Pre, lsit2D_Pre = [], [], [], []
        for i in range(input_size-1):
            S_base = s_obs[i, 0]
            I_base = i_obs[i, 0]
            R_base = r_obs[i, 0]
            D_base = d_obs[i, 0]
            beta_param = scale2beta*nn_beta[i, 0]
            gamma_param = scale2gamma*nn_gamma[i, 0]
            mu_param = scale2mu*nn_mu[i, 0]
            s_update, i_update, r_update, d_update = RK4_SIRD.SIRD_RK4(
                t=t[i], s0=S_base, i0=I_base, r0=R_base, d0=D_base, h=1.0,
                beta=beta_param, gamma=gamma_param, mu=mu_param)

            lsit2S_Pre.append(torch.reshape(s_update, shape=(1, 1)))
            lsit2I_Pre.append(torch.reshape(i_update, shape=(1, 1)))
            lsit2R_Pre.append(torch.reshape(r_update, shape=(1, 1)))
            lsit2D_Pre.append(torch.reshape(d_update, shape=(1, 1)))

        s_pre = torch.concat(lsit2S_Pre, dim=0)
        i_pre = torch.concat(lsit2I_Pre, dim=0)
        r_pre = torch.concat(lsit2R_Pre, dim=0)
        d_pre = torch.concat(lsit2D_Pre, dim=0)

        # ttt=torch.reshape(s_obs[1: input_size, 0], shape=[-1, 1])

        diff2S = torch.reshape(s_obs[1: input_size, 0], shape=[-1, 1]) - s_pre
        diff2I = torch.reshape(i_obs[1: input_size, 0], shape=[-1, 1]) - i_pre
        diff2R = torch.reshape(r_obs[1: input_size, 0], shape=[-1, 1]) - r_pre
        diff2D = torch.reshape(d_obs[1: input_size, 0], shape=[-1, 1]) - d_pre

        if str.lower(loss_type) == 'l2_loss':
            Loss2S = torch.mean(torch.square(diff2S))
            Loss2I = torch.mean(torch.square(diff2I))
            Loss2R = torch.mean(torch.square(diff2R))
            Loss2D = torch.mean(torch.square(diff2D))
        elif str.lower(loss_type) == 'lncosh_loss':
            Loss2S = (1/scale2lncosh)*torch.mean(torch.log(torch.cosh(scale2lncosh*diff2S)))
            Loss2I = (1/scale2lncosh)*torch.mean(torch.log(torch.cosh(scale2lncosh*diff2I)))
            Loss2R = (1/scale2lncosh)*torch.mean(torch.log(torch.cosh(scale2lncosh*diff2R)))
            Loss2D = (1/scale2lncosh)*torch.mean(torch.log(torch.cosh(scale2lncosh*diff2D)))

        return ParamsNN, Loss2S, Loss2I, Loss2R, Loss2D

    def get_regularSum2WB(self):
        sum_WB2Beta = self.BetaNN.get_regular_sum2WB(regular_model=self.opt2regular_WB)
        sum_WB2Gamma = self.GammaNN.get_regular_sum2WB(regular_model=self.opt2regular_WB)
        sum_WB2Mu = self.MuNN.get_regular_sum2WB(regular_model=self.opt2regular_WB)
        return sum_WB2Beta + sum_WB2Gamma + sum_WB2Mu

    def evaluate_RK4DNN(self, t=None, s_init=10.0, i_init=10.0, r_init=10.0, d_init=10.0, size2predict=7,
                        scale2beta=1.0, scale2gamma=0.1, scale2mu=0.05):
        assert (t is not None)  # 该处的t是训练过程的时间， size2predict 是测试的规模大小
        shape2t = t.shape
        lenght2t_shape = len(shape2t)
        assert (lenght2t_shape == 2)
        assert (shape2t[-1] == 1)

        nn2beta = self.BetaNN(t, scale=self.factor2freq, sFourier=self.sFourier)
        nn2gamma = self.GammaNN(t, scale=self.factor2freq, sFourier=self.sFourier)
        nn2mu = self.MuNN(t, scale=self.factor2freq, sFourier=self.sFourier)
        ParamsNN = torch.concat([nn2beta, nn2gamma, nn2mu], dim=-1)

        s_base = s_init
        i_base = i_init
        r_base = r_init
        d_base = d_init

        lsit2S_Pre, lsit2I_Pre, lsit2R_Pre, lsit2D_Pre = [], [], [], []

        for i in range(size2predict):
            nn_beta = scale2beta*nn2beta[i, 0]
            nn_gamma = scale2gamma*nn2gamma[i, 0]
            nn_mu = scale2mu*nn2mu[i, 0]
            s_update, i_update, r_update, d_update = RK4_SIRD.SIRD_RK4(
                t=t[i], s0=s_base, i0=i_base, r0=r_base, d0=d_base, h=1.0, beta=nn_beta, gamma=nn_gamma, mu=nn_mu)
            s_base = s_update
            i_base = i_update
            r_base = r_update
            d_base = d_update

            lsit2S_Pre.append(torch.reshape(s_update, shape=(1, 1)))
            lsit2I_Pre.append(torch.reshape(i_update, shape=(1, 1)))
            lsit2R_Pre.append(torch.reshape(r_update, shape=(1, 1)))
            lsit2D_Pre.append(torch.reshape(d_update, shape=(1, 1)))

        S_Pre = torch.concat(lsit2S_Pre, dim=0)
        I_Pre = torch.concat(lsit2I_Pre, dim=0)
        R_Pre = torch.concat(lsit2R_Pre, dim=0)
        D_Pre = torch.concat(lsit2D_Pre, dim=0)

        return ParamsNN, S_Pre, I_Pre, R_Pre, D_Pre

    def evaluate_RK4DNN_FixedParas(self, t=None, s_init=10.0, i_init=10.0, r_init=10.0, d_init=10.0, size2predict=7,
                                   opt2fixed_paras='last2train', mean2para=3, scale2beta=1.0, scale2gamma=0.1,
                                   scale2mu=0.05):
        assert (t is not None)
        shape2t = t.shape
        lenght2t_shape = len(shape2t)
        assert (lenght2t_shape == 2)
        assert (shape2t[-1] == 1)

        nn2beta = self.BetaNN(t, scale=self.factor2freq, sFourier=self.sFourier)
        nn2gamma = self.GammaNN(t, scale=self.factor2freq, sFourier=self.sFourier)
        nn2mu = self.MuNN(t, scale=self.factor2freq, sFourier=self.sFourier)
        ParamsNN = torch.concat([nn2beta, nn2gamma, nn2mu], dim=-1)

        # 训练过程中最后一天的参数作为固定参数
        if opt2fixed_paras == 'last2train':
            nn_beta = scale2beta*nn2beta[0, 0]
            nn_gamma = scale2gamma*nn2gamma[0, 0]
            nn_mu = scale2mu*nn2mu[0, 0]
        else:  # 训练过程中最后几天的参数的均值作为固定参数，如三天的参数均值作为固定参数
            nn_beta = scale2beta*torch.mean(nn2beta, dim=0)
            nn_gamma = scale2gamma*torch.mean(nn2gamma, dim=0)
            nn_mu = scale2mu*torch.mean(nn2mu, dim=0)

        s_base = s_init
        i_base = i_init
        r_base = r_init
        d_base = d_init

        lsit2S_Pre, lsit2I_Pre, lsit2R_Pre, lsit2D_Pre = [], [], [], []

        for i in range(size2predict):
            s_update, i_update, r_update, d_update = RK4_SIRD.SIRD_RK4(
                t=t+i, s0=s_base, i0=i_base, r0=r_base, d0=d_base, h=1.0, beta=nn_beta, gamma=nn_gamma, mu=nn_mu)
            s_base = s_update
            i_base = i_update
            r_base = r_update
            d_base = d_update

            lsit2S_Pre.append(torch.reshape(s_update, shape=(1, 1)))
            lsit2I_Pre.append(torch.reshape(i_update, shape=(1, 1)))
            lsit2R_Pre.append(torch.reshape(r_update, shape=(1, 1)))
            lsit2D_Pre.append(torch.reshape(d_update, shape=(1, 1)))

        S_Pre = torch.concat(lsit2S_Pre, dim=0)
        I_Pre = torch.concat(lsit2I_Pre, dim=0)
        R_Pre = torch.concat(lsit2R_Pre, dim=0)
        D_Pre = torch.concat(lsit2D_Pre, dim=0)

        return ParamsNN, S_Pre, I_Pre, R_Pre, D_Pre


def solve_SIRD(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log2train', R['name2act_hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    logUtils.dictionary_out2file2RK4(R, log_fileout)

    # 问题需要的设置
    trainSet_szie = R['size2train']          # 训练集大小,给定一个数据集，拆分训练集和测试集时，需要多大规模的训练集
    batchsize2train = R['batch_size2train']
    batchsize2test = R['batch_size2test']
    penalty2WB = R['penalty2weight_biases']  # Regularization parameter for weights and biases
    init_lr = R['learning_rate']
    act_func = R['name2act_hidden']

    SIRDmodel = RK4DNN(input_dim=R['input_dim'], out_dim=R['output_dim'], hidden_layer=R['hidden_layers'],
                       Model_name=R['model2NN'], name2actIn=R['name2act_in'], name2actHidden=R['name2act_hidden'],
                       name2actOut=R['name2act_out'], opt2regular_WB=R['regular_wb_model'], type2numeric='float32',
                       factor2freq=R['freq'], sFourier=R['sfourier'], repeat_highFreq=R['if_repeat_High_freq'],
                       use_gpu=R['use_gpu'], No2GPU=R['gpuNo'])

    if True == R['use_gpu']:
        SIRDmodel = SIRDmodel.cuda(device='cuda:'+str(R['gpuNo']))

    params2Beta_Net = SIRDmodel.BetaNN.parameters()
    params2Gamma_Net = SIRDmodel.GammaNN.parameters()
    params2Mu_Net = SIRDmodel.MuNN.parameters()

    params2Net = itertools.chain(params2Beta_Net, params2Gamma_Net, params2Mu_Net)

    # 定义优化方法，并给定初始学习率
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr)                     # SGD
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr, momentum=0.8)       # momentum
    # optimizer = torch.optim.RMSprop(params2Net, lr=init_lr, alpha=0.95)     # RMSProp
    optimizer = torch.optim.Adam(params2Net, lr=init_lr)                      # Adam

    # 定义更新学习率的方法
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.99)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.975)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.975)

    # filename = 'data2csv/Wuhan.csv'
    # filename = 'data2csv/Italia_data.csv'
    # filename = 'data2csv/Korea_data.csv'
    # filename = 'data2csv/minnesota.csv'
    # filename = 'data2csv/minnesota2.csv'
    filename = 'data/minnesota3.csv'

    # 根据文件读入数据，然后存放在 numpy 数组里面
    date, data2S, data2I, data2R, data2D = dataUtils.load_4csvData_cal_S(
        datafile=filename, total_population=R['total_population'])

    assert (trainSet_szie + batchsize2test <= len(data2I))
    if R['normalize_population'] == 1:
        # 不归一化数据
        train_date, train_data2s, train_data2i, train_data2r, train_data2d, test_date, test_data2s, test_data2i, \
        test_data2r, test_data2d = dataUtils.split_5csvData2train_test(
            date, data2S, data2I, data2R, data2D, size2train=trainSet_szie, normalFactor=1.0, to_torch=True,
            to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'], use_grad2x=True)
    elif (R['total_population'] != R['normalize_population']) and R['normalize_population'] != 1:
        # 归一化数据，使用的归一化数值小于总“人口”
        train_date, train_data2s, train_data2i, train_data2r, train_data2d, test_date, test_data2s, test_data2i, \
        test_data2r, test_data2d = dataUtils.split_5csvData2train_test(
            date, data2S, data2I, data2R, data2D, size2train=trainSet_szie, normalFactor=R['normalize_population'],
            to_torch=True, to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'], use_grad2x=True)
    elif (R['total_population'] == R['normalize_population']) and R['normalize_population'] != 1:
        # 归一化数据，使用总“人口”归一化数据
        train_date, train_data2s, train_data2i, train_data2r, train_data2d, test_date, test_data2s, test_data2i, \
        test_data2r, test_data2d = dataUtils.split_5csvData2train_test(
            date, data2S, data2I, data2R, data2D, size2train=trainSet_szie, normalFactor=R['total_population'],
            to_torch=True, to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'], use_grad2x=True)

    # 对于时间数据来说，验证模型的合理性，要用连续的时间数据验证.
    test_t_bach = dataUtils.sample_testDays_serially(test_date, batchsize2test, is_torch=True)

    # 由于将数据拆分为训练数据和测试数据时，进行了归一化处理，故这里不用归一化
    s_obs_test = dataUtils.sample_testData_serially(test_data2s, batchsize2test, normalFactor=1.0, is_torch=True)
    i_obs_test = dataUtils.sample_testData_serially(test_data2i, batchsize2test, normalFactor=1.0, is_torch=True)
    r_obs_test = dataUtils.sample_testData_serially(test_data2r, batchsize2test, normalFactor=1.0, is_torch=True)
    d_obs_test = dataUtils.sample_testData_serially(test_data2d, batchsize2test, normalFactor=1.0, is_torch=True)

    # 测试过程的初始值，选为训练集的最后一天的值
    init2S_test = train_data2s[trainSet_szie - 1]
    init2I_test = train_data2i[trainSet_szie - 1]
    init2R_test = train_data2r[trainSet_szie - 1]
    init2D_test = train_data2d[trainSet_szie - 1]

    # 将训练集的最后一天和测试集的前n天连接起来，作为新的测试批大小
    last_train_ts = torch.reshape(train_date[trainSet_szie - 5:-1], shape=[-1, 1])
    last_train_t = torch.reshape(train_date[trainSet_szie - 1], shape=[1, 1])
    new_test_t_bach = torch.concat([last_train_t, torch.reshape(test_t_bach[0:-1, 0], shape=[-1, 1])], dim=0)

    t0 = time.time()
    loss_all, loss_s_all, loss_i_all, loss_r_all, loss_d_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_epoch = []
    test_mse2s_all, test_mse2i_all, test_mse2r_all, test_mse2d_all = [], [], [], []
    test_rel2s_all, test_rel2i_all, test_rel2r_all, test_rel2d_all = [], [], [], []

    test_mse2s_Fix_all, test_mse2i_Fix_all, test_mse2r_Fix_all, test_mse2d_Fix_all = [], [], [], []
    test_rel2s_Fix_all, test_rel2i_Fix_all, test_rel2r_Fix_all, test_rel2d_Fix_all = [], [], [], []
    for epoch in range(R['max_epoch'] + 1):
        if batchsize2train == trainSet_szie:
            t_batch = torch.reshape(train_date, shape=[-1, 1])
            s_obs = torch.reshape(train_data2s, shape=[-1, 1])
            i_obs = torch.reshape(train_data2i, shape=[-1, 1])
            r_obs = torch.reshape(train_data2r, shape=[-1, 1])
            d_obs = torch.reshape(train_data2d, shape=[-1, 1])
        else:
            t_batch, s_obs, i_obs, r_obs, d_obs = \
                dataUtils.randSample_Normalize_5existData(
                    train_date, train_data2s, train_data2i, train_data2r, train_data2d, batchsize=batchsize2train,
                    normalFactor=1.0, sampling_opt=R['opt2sample'])

        params, loss2s, loss2i, loss2r, loss2d = SIRDmodel.RK4Iteration(
            t=t_batch, input_size=batchsize2train, s_obs=s_obs, i_obs=i_obs, r_obs=r_obs,
            d_obs=d_obs, loss_type=R['loss_type'], scale2lncosh=R['lambda2lncosh'], scale2beta=R['scale_modify2beta'],
            scale2gamma=R['scale_modify2gamma'], scale2mu=R['scale_modify2mu'])

        regularSum2WB = SIRDmodel.get_regularSum2WB()
        Paras_PWB = penalty2WB * regularSum2WB

        # loss = loss2s + loss2i + loss2r + loss2d + Paras_PWB

        # loss = loss2s + loss2i + 10*loss2r + 20*loss2d + Paras_PWB
        # loss = loss2s + loss2i + 5 * loss2r + 10 * loss2d + Paras_PWB
        loss = loss2s + loss2i + 10 * loss2r + 20 * loss2d + Paras_PWB

        loss_s_all.append(loss2s.item())
        loss_i_all.append(loss2i.item())
        loss_r_all.append(loss2r.item())
        loss_d_all.append(loss2d.item())
        loss_all.append(loss.item())

        optimizer.zero_grad()             # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward(retain_graph=True)  # 对loss关于Ws和Bs求偏导
        optimizer.step()                  # 更新参数Ws和Bs
        scheduler.step()

        if epoch % 100 == 0:
            run_times = time.time() - t0
            tmp_lr = optimizer.param_groups[0]['lr']
            logUtils.print_training2OneNet(epoch, run_times, tmp_lr, Paras_PWB.item(), loss2s.item(), loss2i.item(),
                                           loss2r.item(), loss2d.item(), loss.item(), log_out=log_fileout)

            # ---------------------------   test network ----------------------------------------------
            test_epoch.append(epoch / 1000)
            paras_nn, S_predict, I_predict, R_predict, D_predict = SIRDmodel.evaluate_RK4DNN(
                t=new_test_t_bach, s_init=init2S_test, i_init=init2I_test, r_init=init2R_test, d_init=init2D_test,
                size2predict=batchsize2test, scale2beta=R['scale_modify2beta'], scale2gamma=R['scale_modify2gamma'],
                scale2mu=R['scale_modify2mu'])

            test_mse2S = torch.mean(torch.square(s_obs_test - S_predict))
            test_mse2I = torch.mean(torch.square(i_obs_test - I_predict))
            test_mse2R = torch.mean(torch.square(r_obs_test - R_predict))
            test_mse2D = torch.mean(torch.square(d_obs_test - D_predict))

            test_rel2S = test_mse2S / torch.mean(torch.square(s_obs_test))
            test_rel2I = test_mse2I / torch.mean(torch.square(i_obs_test))
            test_rel2R = test_mse2R / torch.mean(torch.square(r_obs_test))
            test_rel2D = test_mse2D / torch.mean(torch.square(d_obs_test))

            test_mse2s_all.append(test_mse2S.item())
            test_mse2i_all.append(test_mse2I.item())
            test_mse2r_all.append(test_mse2R.item())
            test_mse2d_all.append(test_mse2D.item())

            test_rel2s_all.append(test_rel2S.item())
            test_rel2i_all.append(test_rel2I.item())
            test_rel2r_all.append(test_rel2R.item())
            test_rel2d_all.append(test_rel2D.item())
            logUtils.print_test2OneNet(test_mse2S.item(), test_mse2I.item(), test_mse2R.item(), test_mse2D.item(),
                                       test_rel2S.item(), test_rel2I.item(), test_rel2R.item(), test_rel2D.item(),
                                       log_out=log_fileout)

            fix_paras_nn, S_predict2fix, I_predict2fix, R_predict2fix, D_predict2fix = \
                SIRDmodel.evaluate_RK4DNN_FixedParas(t=last_train_t, s_init=init2S_test, i_init=init2I_test,
                                                     r_init=init2R_test, d_init=init2D_test,
                                                     size2predict=batchsize2test,
                                                     opt2fixed_paras='last2train', scale2beta=R['scale_modify2beta'],
                                                     scale2gamma=R['scale_modify2gamma'], scale2mu=R['scale_modify2mu'])

            test_mse2S_fix = torch.mean(torch.square(s_obs_test - S_predict2fix))
            test_mse2I_fix = torch.mean(torch.square(i_obs_test - I_predict2fix))
            test_mse2R_fix = torch.mean(torch.square(r_obs_test - R_predict2fix))
            test_mse2D_fix = torch.mean(torch.square(d_obs_test - D_predict2fix))

            test_rel2S_fix = test_mse2S_fix / torch.mean(torch.square(s_obs_test))
            test_rel2I_fix = test_mse2I_fix / torch.mean(torch.square(i_obs_test))
            test_rel2R_fix = test_mse2R_fix / torch.mean(torch.square(r_obs_test))
            test_rel2D_fix = test_mse2D_fix / torch.mean(torch.square(d_obs_test))

            test_mse2s_Fix_all.append(test_mse2S_fix.item())
            test_mse2i_Fix_all.append(test_mse2I_fix.item())
            test_mse2r_Fix_all.append(test_mse2R_fix.item())
            test_mse2d_Fix_all.append(test_mse2D_fix.item())

            test_rel2s_Fix_all.append(test_rel2S_fix.item())
            test_rel2i_Fix_all.append(test_rel2I_fix.item())
            test_rel2r_Fix_all.append(test_rel2R_fix.item())
            test_rel2d_Fix_all.append(test_rel2D_fix.item())

            logUtils.print_testFix_paras2OneNet(test_mse2S_fix.item(), test_mse2I_fix.item(), test_mse2R_fix.item(),
                                                test_mse2D_fix.item(), test_rel2S_fix.item(), test_rel2I_fix.item(),
                                                test_rel2R_fix.item(), test_rel2D_fix.item(), log_out=log_fileout)

    # ------------------- save the training results into mat file and plot them -------------------------
    saveData.save_SIRD_trainLoss2mat_no_N(loss_s_all, loss_i_all, loss_r_all, loss_d_all, actName=act_func,
                                          outPath=R['FolderName'])

    plotData.plotTrain_loss_1act_func(loss_s_all, lossType='loss2s', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_i_all, lossType='loss2i', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_r_all, lossType='loss2r', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_d_all, lossType='loss2d', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)

    # ------------------- save the testing results into mat file and plot them -------------------------
    plotData.plotTest_MSE_REL(test_mse2s_all, test_rel2s_all, test_epoch, actName='S', seedNo=R['seed'],
                              outPath=R['FolderName'], xaxis_scale=False, yaxis_scale=True)
    plotData.plotTest_MSE_REL(test_mse2i_all, test_rel2i_all, test_epoch, actName='I', seedNo=R['seed'],
                              outPath=R['FolderName'], xaxis_scale=False, yaxis_scale=True)
    plotData.plotTest_MSE_REL(test_mse2r_all, test_rel2r_all, test_epoch, actName='R', seedNo=R['seed'],
                              outPath=R['FolderName'], xaxis_scale=False, yaxis_scale=True)
    plotData.plotTest_MSE_REL(test_mse2d_all, test_rel2d_all, test_epoch, actName='D', seedNo=R['seed'],
                              outPath=R['FolderName'], xaxis_scale=False, yaxis_scale=True)

    plotData.plotTest_MSE_REL(test_mse2s_Fix_all, test_rel2s_Fix_all, test_epoch, actName='S_Fix', seedNo=R['seed'],
                              outPath=R['FolderName'], xaxis_scale=False, yaxis_scale=True)
    plotData.plotTest_MSE_REL(test_mse2i_Fix_all, test_rel2i_Fix_all, test_epoch, actName='I_Fix', seedNo=R['seed'],
                              outPath=R['FolderName'], xaxis_scale=False, yaxis_scale=True)
    plotData.plotTest_MSE_REL(test_mse2r_Fix_all, test_rel2r_Fix_all, test_epoch, actName='R_Fix', seedNo=R['seed'],
                              outPath=R['FolderName'], xaxis_scale=False, yaxis_scale=True)
    plotData.plotTest_MSE_REL(test_mse2d_Fix_all, test_rel2d_Fix_all, test_epoch, actName='D_Fix', seedNo=R['seed'],
                              outPath=R['FolderName'], xaxis_scale=False, yaxis_scale=True)

    if True == R['use_gpu']:
        # print('********************** with gpu *****************************')
        test_t_bach_numpy = test_t_bach.cpu().detach().numpy()
        s_obs_test_numpy = s_obs_test.cpu().detach().numpy()
        i_obs_test_numpy = i_obs_test.cpu().detach().numpy()
        r_obs_test_numpy = r_obs_test.cpu().detach().numpy()
        d_obs_test_numpy = d_obs_test.cpu().detach().numpy()

        s_pre_test_numpy = S_predict.cpu().detach().numpy()
        i_pre_test_numpy = I_predict.cpu().detach().numpy()
        r_pre_test_numpy = R_predict.cpu().detach().numpy()
        d_pre_test_numpy = D_predict.cpu().detach().numpy()

        s_fix_test_numpy = S_predict2fix.cpu().detach().numpy()
        i_fix_test_numpy = I_predict2fix.cpu().detach().numpy()
        r_fix_test_numpy = R_predict2fix.cpu().detach().numpy()
        d_fix_test_numpy = D_predict2fix.cpu().detach().numpy()
    else:
        # print('********************* without gpu **********************')
        test_t_bach_numpy = test_t_bach.detach().numpy()

        s_obs_test_numpy = s_obs_test.detach().numpy()
        i_obs_test_numpy = i_obs_test.detach().numpy()
        r_obs_test_numpy = r_obs_test.detach().numpy()
        d_obs_test_numpy = d_obs_test.detach().numpy()

        s_pre_test_numpy = S_predict.detach().numpy()
        i_pre_test_numpy = I_predict.detach().numpy()
        r_pre_test_numpy = R_predict.detach().numpy()
        d_pre_test_numpy = D_predict.detach().numpy()

        s_fix_test_numpy = S_predict2fix.detach().numpy()
        i_fix_test_numpy = I_predict2fix.detach().numpy()
        r_fix_test_numpy = R_predict2fix.detach().numpy()
        d_fix_test_numpy = D_predict2fix.detach().numpy()

    plotData.plot_3solus2SIRD_test(s_obs_test_numpy, s_pre_test_numpy, s_fix_test_numpy, exact_name='S_true',
                                   solu1_name='S_pre2time', solu2_name='S_pre2fix',  file_name='S_solu',
                                   coord_points=test_t_bach_numpy, outPath=R['FolderName'])
    plotData.plot_3solus2SIRD_test(i_obs_test_numpy, i_pre_test_numpy, i_fix_test_numpy, exact_name='I_true',
                                   solu1_name='I_pre2time', solu2_name='I_pre2fix', file_name='I_solu',
                                   coord_points=test_t_bach_numpy, outPath=R['FolderName'])
    plotData.plot_3solus2SIRD_test(r_obs_test_numpy, r_pre_test_numpy, r_fix_test_numpy, exact_name='R_true',
                                   solu1_name='R_pre2time', solu2_name='R_pre2fix', file_name='R_solu',
                                   coord_points=test_t_bach_numpy, outPath=R['FolderName'])
    plotData.plot_3solus2SIRD_test(d_obs_test_numpy, d_pre_test_numpy, d_fix_test_numpy, exact_name='D_true',
                                   solu1_name='D_pre2time', solu2_name='D_pre2fix', file_name='D_solu',
                                   coord_points=test_t_bach_numpy, outPath=R['FolderName'])

    saveData.save_SIRD_testSolus2mat(S_predict, I_predict, R_predict, D_predict, name2solus1='S_pre',
                                     name2solus2='I_pre', name2solus3='R_pre', name2solus4='D_pre',
                                     file_name='timeParas', outPath=R['FolderName'])

    saveData.save_SIRD_testSolus2mat(S_predict2fix, I_predict2fix, R_predict2fix, D_predict2fix, name2solus1='S_pre',
                                     name2solus2='I_pre', name2solus3='R_pre', name2solus4='D_pre',
                                     file_name='fixParas', outPath=R['FolderName'])

    saveData.save_SIRD_testParas2mat(paras_nn[:, 0], paras_nn[:, 1], paras_nn[:, 2], name2para1='Beta',
                                     name2para2='Gamma', name2para3='Mu', outPath=R['FolderName'])


if __name__ == "__main__":
    R = {}
    R['gpuNo'] = 0
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 设置当前使用的GPU设备仅为第 0,1,2,3 块GPU, 设备名称为'/gpu:0'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # The path of saving files
    store_file = 'SIRD_RK4'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])  # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Copy and save this file to given path %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if platform.system() == 'Windows':
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    # step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    # R['activate_stop'] = int(step_stop_flag)
    R['activate_stop'] = int(0)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 10000
    # R['max_epoch'] = 20000
    # R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setups of problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    R['input_dim'] = 1  # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1  # 输出维数

    R['ODE_type'] = 'SIRD'
    R['equa_name'] = 'minnesota'

    R['opt2RK4'] = 'rk4_iter'
    # R['opt2RK4'] = 'rk4_vector'

    R['total_population'] = 3450000  # 总的“人口”数量

    # R['normalize_population'] = 3450000                # 归一化时使用的“人口”数值
    R['normalize_population'] = 10000
    # R['normalize_population'] = 5000
    # R['normalize_population'] = 2000
    # R['normalize_population'] = 1000
    # R['normalize_population'] = 1

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    R['size2train'] = 280  # 训练集的大小
    R['batch_size2train'] = 30                        # 训练数据的批大小
    # R['batch_size2train'] = 80                        # 训练数据的批大小
    # R['batch_size2train'] = 280  # 训练数据的批大小
    R['batch_size2test'] = 7       # 训练数据的批大小
    # R['opt2sample'] = 'random_sample'                 # 训练集的选取方式--随机采样
    # R['opt2sample'] = 'rand_sample_sort'              # 训练集的选取方式--随机采样后按时间排序
    R['opt2sample'] = 'windows_rand_sample'  # 训练集的选取方式--随机窗口采样(以随机点为基准，然后滑动窗口采样)

    # The types of loss function
    R['loss_type'] = 'L2_loss'
    # R['loss_type'] = 'lncosh_loss'
    # R['lambda2lncosh'] = 0.01
    R['lambda2lncosh'] = 0.05  # 这个因子效果很好
    # R['lambda2lncosh'] = 0.075
    # R['lambda2lncosh'] = 0.1
    # R['lambda2lncosh'] = 0.5
    # R['lambda2lncosh'] = 1.0
    # R['lambda2lncosh'] = 50.0

    # The options of optimizers, learning rate, the decay of learning rate and the model of training network
    R['optimizer_name'] = 'Adam'  # 优化器

    # R['learning_rate'] = 1e-2                          # 学习率

    R['learning_rate'] = 5e-3  # 学习率

    # R['learning_rate'] = 2e-4                          # 学习率

    R['train_model'] = 'union_training'

    # 正则化权重和偏置的模式
    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'
    # R['penalty2weight_biases'] = 0.000                # Regularization parameter for weights
    R['penalty2weight_biases'] = 0.00001                # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.00005                # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0001               # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0005               # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.001                # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025               # Regularization parameter for weights

    # 边界的惩罚处理方式,以及边界的惩罚因子
    R['activate_penalty2pt_increase'] = 1
    # R['init_penalty2predict_true'] = 1000             # Regularization factor for the  prediction and true
    # R['init_penalty2predict_true'] = 100              # Regularization factor for the  prediction and true
    R['init_penalty2predict_true'] = 10                 # Regularization factor for the  prediction and true

    # &&&&&&& The option fo Network model, the setups of hidden-layers and the option of activation function &&&&&&&&&&&
    # R['model2NN'] = 'DNN'
    # R['model2NN'] = 'Scale_DNN'
    R['model2NN'] = 'Fourier_DNN'

    if R['model2NN'] == 'Fourier_DNN':
        R['hidden_layers'] = (40, 50, 25, 25, 10)
    else:
        R['hidden_layers'] = (90, 50, 25, 25, 10)

    # R['name2act_in'] = 'relu'
    # R['name2act_in'] = 'leaky_relu'
    # R['name2act_in'] = 'elu'
    # R['name2act_in'] = 'gelu'
    # R['name2act_in'] = 'mgelu'
    # R['name2act_in'] = 'tanh'
    # R['name2act_in'] = 'sin'
    R['name2act_in'] = 'sinAddcos'
    # R['name2act_in'] = 's2relu'

    # R['name2act_hidden'] = 'relu'
    # R['name2act_hidden'] = 'tanh'
    # R['name2act_hidden']' = leaky_relu'
    # R['name2act_hidden'] = 'srelu'
    # R['name2act_hidden'] = 's2relu'
    # R['name2act_hidden'] = 'sin'
    R['name2act_hidden'] = 'sinAddcos'
    # R['name2act_hidden'] = 'elu'

    # R['name2act_out'] = 'linear'
    R['name2act_out'] = 'sigmoid'

    # &&&&&&&&&&&&&&&&&&&&& some other factors for network &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    R['freq'] = np.concatenate(([1], np.arange(1, 30 - 1)), axis=0)  # 网络的频率范围设置

    R['if_repeat_High_freq'] = False

    R['sfourier'] = 1.0
    R['use_gpu'] = True

    R['scale_modify2beta'] = 1.0
    R['scale_modify2gamma'] = 0.1
    R['scale_modify2mu'] = 0.01

    solve_SIRD(R)