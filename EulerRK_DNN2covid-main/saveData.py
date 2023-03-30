import scipy.io as scio


def true_value2convid(trueArray, name2Array=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, str.upper(name2Array))
    key2mat = name2Array
    scio.savemat(outFile2data, {key2mat: trueArray})


def save_SIR_trainLoss2mat_no_N(loss_sArray, loss_iArray, loss_rArray, loss_allArray, actName=None, outPath=None):
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'loss_s'
    key2mat_2 = 'loss_i'
    key2mat_3 = 'loss_r'
    key2mat_4 = 'loss_all'
    scio.savemat(outFile2data, {key2mat_1: loss_sArray, key2mat_2: loss_iArray, key2mat_3: loss_rArray,
                                key2mat_4: loss_allArray})


def save_SIR_trainLoss2mat(loss_sArray, loss_iArray, loss_rArray, loss_nArray, actName=None, outPath=None):
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'loss_s'
    key2mat_2 = 'loss_i'
    key2mat_3 = 'loss_r'
    key2mat_4 = 'loss_n'
    scio.savemat(outFile2data, {key2mat_1: loss_sArray, key2mat_2: loss_iArray, key2mat_3: loss_rArray,
                                key2mat_4: loss_nArray})


def save_SIRD_trainLoss2mat_no_N(loss_sArray, loss_iArray, loss_rArray, loss_dArray, actName=None, outPath=None):
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'loss_s'
    key2mat_2 = 'loss_i'
    key2mat_3 = 'loss_r'
    key2mat_4 = 'loss_d'
    scio.savemat(outFile2data, {key2mat_1: loss_sArray, key2mat_2: loss_iArray, key2mat_3: loss_rArray,
                                key2mat_4: loss_dArray})


def save_SIRD_trainLoss2mat(loss_sArray, loss_iArray, loss_rArray, loss_dArray, loss_nArray, actName=None, outPath=None):
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'loss_s'
    key2mat_2 = 'loss_i'
    key2mat_3 = 'loss_r'
    key2mat_4 = 'loss_d'
    key2mat_5 = 'loss_n'
    scio.savemat(outFile2data, {key2mat_1: loss_sArray, key2mat_2: loss_iArray, key2mat_3: loss_rArray,
                                key2mat_4: loss_dArray, key2mat_5: loss_nArray})


def save_SEIR_trainLoss2mat(loss_sArray, loss_eArray, loss_iArray, loss_rArray, loss_nArray, actName=None,
                            outPath=None):
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'loss_s'
    key2mat_2 = 'loss_e'
    key2mat_3 = 'loss_i'
    key2mat_4 = 'loss_r'
    key2mat_5 = 'loss_n'
    scio.savemat(outFile2data, {key2mat_1: loss_sArray, key2mat_2: loss_eArray, key2mat_3: loss_iArray,
                                key2mat_4: loss_rArray, key2mat_5: loss_nArray})


def save_SEIRD_trainLoss2mat(loss2s_arr, loss2e_arr, loss2i_arr, loss2r_arr, loss2d_arr, loss2n_arr, actName=None,
                             outPath=None):
    outFile2data = '%s/LossSEIRD2%s.mat' % (outPath, actName)
    key2mat_1 = 'loss_s'
    key2mat_2 = 'loss_e'
    key2mat_3 = 'loss_i'
    key2mat_4 = 'loss_r'
    key2mat_5 = 'loss_d'
    key2mat_6 = 'loss_n'
    scio.savemat(outFile2data, {key2mat_1: loss2s_arr, key2mat_2: loss2e_arr, key2mat_3: loss2i_arr,
                                key2mat_4: loss2r_arr, key2mat_5: loss2d_arr, key2mat_6: loss2n_arr})


def save_trainSolu2mat_Covid(solus_array, name2solus=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, name2solus)
    key2mat_1 = str(name2solus)
    scio.savemat(outFile2data, {key2mat_1: solus_array})


def save_trainParas2mat_Covid(paras_array, name2para=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, name2para)
    key2mat_1 = str(name2para)
    scio.savemat(outFile2data, {key2mat_1: paras_array})


def save_test_paras2mat_Covid(para_array, name2para=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, name2para)
    key2mat_1 = str(name2para)
    scio.savemat(outFile2data, {key2mat_1: para_array})


def save_SIR_testSolus2mat(solu1_array, solu2_array, solu3_array, name2solus1=None,
                           name2solus2=None, name2solus3=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, 'solus2test')
    key2mat_1 = str(name2solus1)
    key2mat_2 = str(name2solus2)
    key2mat_3 = str(name2solus3)
    scio.savemat(outFile2data, {key2mat_1: solu1_array, key2mat_2: solu2_array, key2mat_3: solu3_array})


def save_SIR_testParas2mat(para1_array, para2_array, name2para1=None, name2para2=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, 'paras2test')
    key2mat_1 = str(name2para1)
    key2mat_2 = str(name2para2)
    scio.savemat(outFile2data, {key2mat_1: para1_array, key2mat_2: para2_array})


def save_SIRD_testSolus2mat(solu1_array, solu2_array, solu3_array, solu4_array, name2solus1=None,
                            name2solus2=None, name2solus3=None, name2solus4=None, file_name=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, 'solu2test_'+str(file_name))
    key2mat_1 = str(name2solus1)
    key2mat_2 = str(name2solus2)
    key2mat_3 = str(name2solus3)
    key2mat_4 = str(name2solus4)
    scio.savemat(outFile2data, {key2mat_1: solu1_array, key2mat_2: solu2_array, key2mat_3: solu3_array,
                                key2mat_4: solu4_array})


def save_SIRD_testParas2mat(para1_array, para2_array, para3_array, name2para1=None, name2para2=None,
                            name2para3=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, 'paras2test')
    key2mat_1 = str(name2para1)
    key2mat_2 = str(name2para2)
    key2mat_3 = str(name2para3)
    scio.savemat(outFile2data, {key2mat_1: para1_array, key2mat_2: para2_array, key2mat_3: para3_array})


def save_SEIR_testSolus2mat(solu1_array, solu2_array, solu3_array, solu4_array, name2solus1=None,
                            name2solus2=None, name2solus3=None, name2solus4=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, 'solus2test')
    key2mat_1 = str(name2solus1)
    key2mat_2 = str(name2solus2)
    key2mat_3 = str(name2solus3)
    key2mat_4 = str(name2solus4)
    scio.savemat(outFile2data, {key2mat_1: solu1_array, key2mat_2: solu2_array, key2mat_3: solu3_array,
                                key2mat_4: solu4_array})


def save_SEIR_testParas2mat(para1_array, para2_array, para3_array, name2para1=None, name2para2=None,
                            name2para3=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, 'paras2test')
    key2mat_1 = str(name2para1)
    key2mat_2 = str(name2para2)
    key2mat_3 = str(name2para3)
    scio.savemat(outFile2data, {key2mat_1: para1_array, key2mat_2: para2_array, key2mat_3: para3_array})


def save_SEIRD_testSolus2mat(solu1_array, solu2_array, solu3_array, solu4_array, solu5_array, name2solus1=None,
                                 name2solus2=None, name2solus3=None, name2solus4=None, name2solus5=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, 'solus2test')
    key2mat_1 = str(name2solus1)
    key2mat_2 = str(name2solus2)
    key2mat_3 = str(name2solus3)
    key2mat_4 = str(name2solus4)
    key2mat_5 = str(name2solus5)
    scio.savemat(outFile2data, {key2mat_1: solu1_array, key2mat_2: solu2_array, key2mat_3: solu3_array,
                                key2mat_4: solu4_array, key2mat_5: solu5_array})


def save_SEIRD_testParas2mat(para1_array, para2_array, para3_array, para4_array, para5_array, para6_array,
                             name2para1=None, name2para2=None, name2para3=None, name2para4=None, name2para5=None,
                             name2para6=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, 'paras2test')
    key2mat_1 = str(name2para1)
    key2mat_2 = str(name2para2)
    key2mat_3 = str(name2para3)
    key2mat_4 = str(name2para4)
    key2mat_5 = str(name2para5)
    key2mat_6 = str(name2para6)
    scio.savemat(outFile2data, {key2mat_1: para1_array, key2mat_2: para2_array, key2mat_3: para3_array,
                                key2mat_4: para4_array, key2mat_5: para5_array, key2mat_6: para6_array})


def save_train_MSE_REL2mat(Mse_data, Rel_data, actName=None, outPath=None):
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'mse'
    key2mat_2 = 'rel'
    scio.savemat(outFile2data, {key2mat_1: Mse_data, key2mat_2: Rel_data})


def save_testMSE_REL2mat(Mse_data, Rel_data, actName=None, outPath=None):
    outFile2data = '%s/test_Err2%s.mat' % (outPath, actName)
    key2mat_1 = 'mse'
    key2mat_2 = 'rel'
    scio.savemat(outFile2data, {key2mat_1: Mse_data, key2mat_2: Rel_data})

