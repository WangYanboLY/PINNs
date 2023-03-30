import numpy as np
import csv
import torch
import pandas as pd


# 使用 Pandas 读取 csv文件
def load_csvdata(fileName, N=3450000, filter_size=5, normalizeFlag=False):
    '''
    fileName: path of csv file
    N: total population in the city(country)
    normalizeFlag: whether normalize the data to 0-1.y
    '''
    df = pd.read_csv(fileName)
    # date_list = np.array(df['date'])
    # date_list = [(i+1) for i in range(len(df))]
    df['current_confiremed'] = df['cum_confirmed'] - df['recovered'] - df['death']  # 对应公式里的I
    # df['pops'] = N - df['current_confiremed'] - df['recovered'] - df['death']
    df['pops'] = N - df['cum_confirmed']

    ## 用filter_size 天的平均值把数据磨平一下，用rolling函数实现
    ## rolling函数的作用是协助着把连续的window个值操作一下
    if filter_size != 0:
        df['pops'] = df['pops'].rolling(window=filter_size).mean()
        df['current_confiremed'] = df['current_confiremed'].rolling(window=filter_size).mean()
        df['recovered'] = df['recovered'].rolling(window=filter_size).mean()
        df['death'] = df['death'].rolling(window=filter_size).mean()

    df = df.dropna(axis=0)

    susceptible_list = np.array(df['pops'])
    infective_list = np.array(df['current_confiremed'])
    recovery_list = np.array(df['recovered'])
    death_list = np.array(df['death'])

    date_list = [(i + 1) for i in range(len(death_list))]

    data_list = [date_list, susceptible_list, infective_list, recovery_list, death_list]

    if normalizeFlag == True:
        data_list = [[item / int(N) for item in sublist] for sublist in data_list]

    return data_list


# 拆分数据为训练集和测试集
def split_data(data_list, train_size=0.75):
    date_list, susceptible_list, infective_list, recovery_list, death_list, *_ = data_list
    train_size = int(len(date_list) * train_size)
    # test = len(date_list) - train_size

    date_train = date_list[0:train_size]
    susceptible_train = susceptible_list[0:train_size]
    infective_train = infective_list[0: train_size]
    recovery_train = recovery_list[0:train_size]
    death_train = death_list[0:train_size]

    date_test = date_list[train_size:-1]
    susceptible_test = susceptible_list[train_size:-1]
    infective_test = infective_list[train_size:-1]
    recovery_test = recovery_list[train_size:-1]
    death_test = death_list[train_size:-1]

    train_data = [date_train, susceptible_train, infective_train, recovery_train, death_train]
    test_data = [date_test, susceptible_test, infective_test, recovery_test, death_test]

    return train_data, test_data


def window_sample(data_list, window_size=7, method='sequential_sort'):
    date_data, s_data, i_data,r_data, d_data, *_ = data_list
    data_length = len(date_data)
    assert method in ['random', 'random_sort', 'sequential_sort']
    if method == 'random':
        indexes = np.random.randint(data_length, size=window_size)
    elif method == 'random_sort':
        indexes_temp = np.random.randint(data_length, size=window_size)
        indexes = np.sort(indexes_temp)
    elif method == 'sequential_sort':
        index_base = np.random.randint(data_length - window_size, size=1)
        indexes = np.arange(index_base, index_base + window_size)

    date_window = [date_data[idx] for idx in indexes]
    s_window = [s_data[idx] for idx in indexes]
    i_window = [i_data[idx] for idx in indexes]
    r_window = [r_data[idx] for idx in indexes]
    d_window = [d_data[idx] for idx in indexes]

    date_window = np.expand_dims(date_window, axis=1)
    s_window = np.expand_dims(s_window, axis=1)
    i_window = np.expand_dims(i_window, axis=1)
    r_window = np.expand_dims(r_window, axis=1)
    d_window = np.expand_dims(d_window, axis=1)

    data_window = [date_window, s_window, i_window, r_window, d_window]

    return data_window


def generate_dataset(df, column_name, time_step=7):
    data = df[[column_name]]
    # dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = torch.utils.data.TensorDataset(data)
    dataset = dataset.window(size=time_step, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(time_step))
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))

    return dataset


def windowed_dataset(data_list, time_step=7, batch_size=4):
    # dataset = tf.data.Dataset.from_tensor_slices(data_list)
    dataset = torch.utils.data.TensorDataset(data_list)
    dataset = dataset.window(size=time_step + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(time_step + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset


def sliding_window(data_list, window_size=14, step_time=7):
    # date, s_data, i_data, r_data, d_data, *_ = data_list
    data_list = np.array(data_list)
    length = len(data_list[0]) - window_size
    for idx in range(0, length, step_time):
        start = idx
        end = start + window_size
        window_data = data_list[:, start: end]
        yield window_data


# 从总体数据集中载入部分数据作为训练集
# 窗口滑动采样时，batch size = window_size
def sample_data(date_data, s_data, i_data, r_data, d_data, window_size=1, sampling_opt=None):
    date_temp = list()
    s_temp = list()
    i_temp = list()
    r_temp = list()
    d_temp = list()
    data_length = len(date_data)
    if sampling_opt.lower() == 'random_sample':
        indexes = np.random.randint(data_length, size=window_size)
    elif sampling_opt.lower() == 'rand_sample_sort':
        indexes_temp = np.random.randint(data_length, size=window_size)
        indexes = np.sort(indexes_temp)
    elif sampling_opt.lower() == 'sequential_sort':
        index_base = np.random.randint(data_length - window_size, size=1)
        indexes = np.arange(index_base, index_base + window_size)
    else:
        print('woring!!!!')
    for i_index in indexes:
        date_temp.append(float(date_data[i_index]))
        s_temp.append(float(s_data[i_index]))
        i_temp.append(float(i_data[i_index]))
        r_temp.append(float(r_data[i_index]))
        d_temp.append(float(d_data[i_index]))

    date_samples = np.array(date_temp)
    # data_samples = np.array(data_temp)
    s_samples = np.array(s_temp)
    i_samples = np.array(i_temp)
    r_samples = np.array(r_temp)
    d_samples = np.array(d_temp)
    date_samples = date_samples.reshape(window_size, 1)
    # data_samples = data_samples.reshape(batchsize, 1)
    s_samples = s_samples.reshape(window_size, 1)
    i_samples = i_samples.reshape(window_size, 1)
    r_samples = r_samples.reshape(window_size, 1)
    d_samples = d_samples.reshape(window_size, 1)

    return date_samples, s_samples, i_samples, r_samples, d_samples


# ----------------------------- 下面是我写的数据处理方式 --------------------------------
def load_2csvData(datafile=None):
    csvdata_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            csvdata_list.append(int(dataItem2csv[1]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata = np.array(csvdata_list)
    return csvdate, csvdata


def load_2csvData_cal_S(datafile=None, total_population=100000):
    csvdata2I_list = []
    csvdata2S_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            csvdata2I_list.append(int(dataItem2csv[1]))
            csvdata2S_list.append(int(total_population)-int(dataItem2csv[1]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata2I = np.array(csvdata2I_list)
    csvdata2S = np.array(csvdata2S_list)
    return csvdate, csvdata2I, csvdata2S


def load_3csvData(datafile=None):
    csvdata1_list = []
    csvdata2_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            csvdata1_list.append(int(dataItem2csv[1]))
            csvdata2_list.append(int(dataItem2csv[2]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata1 = np.array(csvdata1_list)
    csvdata2 = np.array(csvdata2_list)
    return csvdate, csvdata1, csvdata2


def load_3csvData_cal_S(datafile=None, total_population=100000):
    csvdata1_list = []
    csvdata2_list = []
    csvdata3_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            csvdata1_list.append(int(dataItem2csv[1]))
            csvdata2_list.append(int(dataItem2csv[2]))
            csvdata3_list.append(int(total_population)-int(dataItem2csv[2]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata1 = np.array(csvdata1_list)
    csvdata2 = np.array(csvdata2_list)
    csvdata3 = np.array(csvdata2_list)
    return csvdate, csvdata1, csvdata2, csvdata3


def load_4csvData(datafile=None):
    csvdata1_list = []
    csvdata2_list = []
    csvdata3_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            csvdata1_list.append(int(dataItem2csv[1]))
            csvdata2_list.append(int(dataItem2csv[2]))
            csvdata3_list.append(int(dataItem2csv[3]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata1 = np.array(csvdata1_list)
    csvdata2 = np.array(csvdata2_list)
    csvdata3 = np.array(csvdata3_list)
    return csvdate, csvdata1, csvdata2, csvdata3


def load_5csvData(datafile=None):
    csvdata1_list = []
    csvdata2_list = []
    csvdata3_list = []
    csvdata4_list = []
    csvdate_list = []
    icount = 0
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            csvdata1_list.append(int(dataItem2csv[1]))
            csvdata2_list.append(int(dataItem2csv[2]))
            csvdata3_list.append(int(dataItem2csv[3]))
            csvdata4_list.append(int(dataItem2csv[4]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata1 = np.array(csvdata1_list)
    csvdata2 = np.array(csvdata2_list)
    csvdata3 = np.array(csvdata3_list)
    csvdata4 = np.array(csvdata4_list)
    return csvdate, csvdata1, csvdata2, csvdata3, csvdata4


def load_4csvData_cal_S(datafile=None, total_population=3450000):
    csvdata2I_list = []
    csvdata2R_list = []
    csvdata2D_list = []
    csvdata2S_list = []
    csvdate_list = []
    icount = 1
    csvreader = csv.reader(open(datafile, 'r'))
    for dataItem2csv in csvreader:
        if str.isnumeric(dataItem2csv[1]):
            # csvdata2I_list.append(int(dataItem2csv[1]))

            csvdata2I_list.append(int(dataItem2csv[1]) - int(dataItem2csv[2]) - int(dataItem2csv[3]))
            csvdata2R_list.append(int(dataItem2csv[2]))
            csvdata2D_list.append(int(dataItem2csv[3]))
            # csvdata2S_list.append(total_population-int(dataItem2csv[1])-int(dataItem2csv[2])-int(dataItem2csv[3]))
            csvdata2S_list.append(total_population - int(dataItem2csv[1]))
            csvdate_list.append(icount)
            icount = icount + 1
    csvdate = np.array(csvdate_list)
    csvdata2I = np.array(csvdata2I_list)
    csvdata2R = np.array(csvdata2R_list)
    csvdata2D = np.array(csvdata2D_list)
    csvdata2S = np.array(csvdata2S_list)
    return csvdate, csvdata2S, csvdata2I, csvdata2R, csvdata2D


# 将数据集拆分为训练集合测试集
def split_2csvData2train_test(date_data, data, size2train=50, normalFactor=10000):

    date2train = date_data[0:size2train]
    data2train = data[0:size2train]/float(normalFactor)

    date2test = date_data[size2train:-1]
    data2test = data[size2train:-1]/float(normalFactor)
    return date2train, data2train, date2test, data2test


# 将数据集拆分为训练集合测试集
def split_3csvData2train_test(date_data, data1, data2, size2train=50, normalFactor=10000):

    date2train = date_data[0:size2train]
    data1_train = data1[0:size2train]/float(normalFactor)
    data2_train = data2[0:size2train] / float(normalFactor)

    date2test = date_data[size2train:-1]
    data1_test = data1[size2train:-1]/float(normalFactor)
    data2_test = data2[size2train:-1] / float(normalFactor)
    return date2train, data1_train, data2_train, date2test, data1_test, data2_test


# 将数据集拆分为训练集合测试集
def split_4csvData2train_test(date_data, data1, data2, data3, size2train=50, normalFactor=1.0):

    date2train = date_data[0:size2train]
    data1_train = data1[0:size2train]/float(normalFactor)
    data2_train = data2[0:size2train] / float(normalFactor)
    data3_train = data3[0:size2train] / float(normalFactor)

    date2test = date_data[size2train:-1]
    data1_test = data1[size2train:-1]/float(normalFactor)
    data2_test = data2[size2train:-1] / float(normalFactor)
    data3_test = data3[size2train:-1] / float(normalFactor)
    return date2train, data1_train, data2_train, data3_train, date2test, data1_test, data2_test, data3_test


# 将数据集拆分为训练集合测试集
def split_5csvData2train_test(date_data, data1, data2, data3, data4, size2train=50, normalFactor=1.0, to_torch=False,
                              to_float=True, to_cuda=False, gpu_no=0, use_grad2x=False):

    date2train = date_data[0:size2train]
    data1_train = data1[0:size2train]/float(normalFactor)
    data2_train = data2[0:size2train] / float(normalFactor)
    data3_train = data3[0:size2train] / float(normalFactor)
    data4_train = data4[0:size2train] / float(normalFactor)

    date2test = date_data[size2train:-1]
    data1_test = data1[size2train:-1]/float(normalFactor)
    data2_test = data2[size2train:-1] / float(normalFactor)
    data3_test = data3[size2train:-1] / float(normalFactor)
    data4_test = data4[size2train:-1] / float(normalFactor)

    if to_float:
        date2train = date2train.astype(np.float32)
        data1_train = data1_train.astype(np.float32)
        data2_train = data2_train.astype(np.float32)
        data3_train = data3_train.astype(np.float32)
        data4_train = data4_train.astype(np.float32)

        date2test = date2test.astype(np.float32)
        data1_test = data1_test.astype(np.float32)
        data2_test = data2_test.astype(np.float32)
        data3_test = data3_test.astype(np.float32)
        data4_test = data4_test.astype(np.float32)

    if to_torch:
        date2train = torch.from_numpy(date2train)
        data1_train = torch.from_numpy(data1_train)
        data2_train = torch.from_numpy(data2_train)
        data3_train = torch.from_numpy(data3_train)
        data4_train = torch.from_numpy(data4_train)

        date2test = torch.from_numpy(date2test)
        data1_test = torch.from_numpy(data1_test)
        data2_test = torch.from_numpy(data2_test)
        data3_test = torch.from_numpy(data3_test)
        data4_test = torch.from_numpy(data4_test)

        if to_cuda:
            date2train = date2train.cuda(device='cuda:' + str(gpu_no))
            data1_train = data1_train.cuda(device='cuda:' + str(gpu_no))
            data2_train = data2_train.cuda(device='cuda:' + str(gpu_no))
            data3_train = data3_train.cuda(device='cuda:' + str(gpu_no))
            data4_train = data4_train.cuda(device='cuda:' + str(gpu_no))

            date2test = date2test.cuda(device='cuda:' + str(gpu_no))
            data1_test = data1_test.cuda(device='cuda:' + str(gpu_no))
            data2_test = data2_test.cuda(device='cuda:' + str(gpu_no))
            data3_test = data3_test.cuda(device='cuda:' + str(gpu_no))
            data4_test = data4_test.cuda(device='cuda:' + str(gpu_no))

        date2train.requires_grad = use_grad2x

    return date2train, data1_train, data2_train, data3_train, data4_train, date2test, data1_test, data2_test, data3_test, data4_test


def randSample_existData(data1, data2, batchsize=1):
    data1_temp = []
    data2_temp = []
    data_length = len(data1)
    indexes = np.random.randint(data_length, size=batchsize)
    for i_index in indexes:
        data1_temp .append(data1[i_index])
        data2_temp .append(data2[i_index])
    data1_samples = np.array(data1_temp)
    data2_samples = np.array(data2_temp)
    data1_samples = data1_samples.reshape(batchsize, 1)
    data2_samples = data2_samples.reshape(batchsize, 1)
    return data1_samples, data2_samples


def randSample_3existData(data1, data2, data3, batchsize=1):
    data1_temp = []
    data2_temp = []
    data3_temp = []
    data_length = len(data1)
    indexes = np.random.randint(data_length, size=batchsize)
    for i_index in indexes:
        data1_temp .append(data1[i_index])
        data2_temp .append(data2[i_index])
        data3_temp.append(data3[i_index])
    data1_samples = np.array(data1_temp)
    data2_samples = np.array(data2_temp)
    data3_samples = np.array(data3_temp)
    data1_samples = data1_samples.reshape(batchsize, 1)
    data2_samples = data2_samples.reshape(batchsize, 1)
    data3_samples = data3_samples.reshape(batchsize, 1)
    return data1_samples, data2_samples, data3_samples


# 从总体数据集中载入部分数据作为训练集
def randSample_Normalize_existData(date_data, data2, batchsize=1, normalFactor=1000, sampling_opt=None):
    date_temp = []
    data_temp = []
    data_length = len(date_data)
    if str.lower(sampling_opt) == 'random_sample':
        indexes = np.random.randint(data_length, size=batchsize)
    elif str.lower(sampling_opt) == 'rand_sample_sort':
        indexes_temp = np.random.randint(data_length, size=batchsize)
        indexes = np.sort(indexes_temp)
    else:
        index_base = np.random.randint(data_length-batchsize, size=1)
        indexes = np.arange(index_base, index_base+batchsize)
    for i_index in indexes:
        date_temp .append(float(date_data[i_index]))
        data_temp .append(float(data2[i_index])/float(normalFactor))
    date_samples = np.array(date_temp)
    data_samples = np.array(data_temp)
    date_samples = date_samples.reshape(batchsize, 1)
    data_samples = data_samples.reshape(batchsize, 1)
    return date_samples, data_samples


# 从总体数据集中载入部分数据作为训练集
def randSample_Normalize_3existData(date_data, data1, data2, batchsize=1, normalFactor=1000, sampling_opt=None):
    date_temp = []
    data1_temp = []
    data2_temp = []
    data_length = len(date_data)
    if str.lower(sampling_opt) == 'random_sample':
        indexes = np.random.randint(data_length, size=batchsize)
    elif str.lower(sampling_opt) == 'rand_sample_sort':
        indexes_temp = np.random.randint(data_length, size=batchsize)
        indexes = np.sort(indexes_temp)
    else:
        index_base = np.random.randint(data_length-batchsize, size=1)
        indexes = np.arange(index_base, index_base+batchsize)
    for i_index in indexes:
        date_temp .append(float(date_data[i_index]))
        data1_temp.append(float(data1[i_index]) / float(normalFactor))
        data2_temp .append(float(data2[i_index])/float(normalFactor))

    date_samples = np.array(date_temp)
    data1_samples = np.array(data1_temp)
    data2_samples = np.array(data2_temp)

    date_samples = date_samples.reshape(batchsize, 1)
    data1_samples = data1_samples.reshape(batchsize, 1)
    data2_samples = data2_samples.reshape(batchsize, 1)
    return date_samples, data1_samples, data2_samples


# 从总体数据集中载入部分数据作为训练集
def randSample_Normalize_5existData(date_data, data1, data2, data3, data4, batchsize=1, normalFactor=1000,
                                    sampling_opt=None, is_torch=False, is_float=True):
    data_length = len(date_data)
    if str.lower(sampling_opt) == 'random_sample':       # 随机采样，然后按时间排序
        indexes = np.random.randint(data_length, size=batchsize)
    elif str.lower(sampling_opt) == 'rand_sample_sort':  # 随机采样，然后按时间排序(或者叫按索引由小到大排序)
        indexes_temp = np.random.randint(data_length, size=batchsize)
        indexes = np.sort(indexes_temp)
    else:  # 窗口滑动
        index_base = np.random.randint(data_length-batchsize, size=1)
        indexes = np.arange(index_base, index_base+batchsize)

    # date_samples = date_data[indexes, :]
    # data1_samples = data1[indexes, :] / float(normalFactor)
    # data2_samples = data2[indexes, :] / float(normalFactor)
    # data3_samples = data3[indexes, :] / float(normalFactor)
    # data4_samples = data4[indexes, :] / float(normalFactor)

    date_samples = date_data[indexes]
    data1_samples = data1[indexes] / float(normalFactor)
    data2_samples = data2[indexes] / float(normalFactor)
    data3_samples = data3[indexes] / float(normalFactor)
    data4_samples = data4[indexes] / float(normalFactor)

    if is_torch:
        date_samples = torch.reshape(date_samples, shape=(batchsize, 1))
        data1_samples = torch.reshape(data1_samples, shape=(batchsize, 1))
        data2_samples = torch.reshape(data2_samples, shape=(batchsize, 1))
        data3_samples = torch.reshape(data3_samples, shape=(batchsize, 1))
        data4_samples = torch.reshape(data4_samples, shape=(batchsize, 1))
    else:
        date_samples = date_samples.reshape(batchsize, 1)
        data1_samples = data1_samples.reshape(batchsize, 1)
        data2_samples = data2_samples.reshape(batchsize, 1)
        data3_samples = data3_samples.reshape(batchsize, 1)
        data4_samples = data4_samples.reshape(batchsize, 1)
    return date_samples, data1_samples, data2_samples, data3_samples, data4_samples


# 对于时间数据来说，验证模型的合理性，要用连续的时间数据验证
def sample_testDays_serially(test_date, batch_size, is_torch=False):
    day_it = test_date[0:batch_size]
    if is_torch:
        day_it = torch.reshape(day_it, shape=(batch_size, 1))
    else:
        day_it = np.reshape(day_it, newshape=(batch_size, 1))
    return day_it


# 对于时间数据来说，验证模型的合理性，要用连续的时间数据验证
def sample_testData_serially(test_data, batch_size, normalFactor=1000, is_torch=False):
    data_it = test_data[0:batch_size]
    data_it = data_it / float(normalFactor)
    if is_torch:
        data_it = torch.reshape(data_it, shape=(batch_size, 1))
    else:
        data_it = np.reshape(data_it, newshape=(batch_size, 1))

    return data_it


# 根据数据序列, 估算每个点处的梯度
def estimate_grad2given_serial_data(serial_data, size2data=100, t_step=1.0, is_torch=False, type2float='float32',
                                    to_torch=False, to_cuda=False, gpu_no=0, use_grad2x=False):
    if is_torch:
        if type2float == 'float32':
            float_type = torch.float32
        elif type2float == 'float64':
            float_type = torch.float64
        elif type2float == 'float16':
            float_type = torch.float16
    else:
        if type2float == 'float32':
            float_type = np.float32
        elif type2float == 'float64':
            float_type = np.float64
        elif type2float == 'float16':
            float_type = np.float16

    if is_torch:
        data2grad = torch.empty([size2data - 4, 1], dtype=float_type)
    else:
        data2grad = np.empty([size2data - 4, 1], dtype=float_type)

    for i in range(size2data-4):
        j = i+2
        d_jminus2 = serial_data[j - 2]
        d_jminus1 = serial_data[j - 1]
        dj = serial_data[j - 1]
        d_jadd1 = serial_data[j + 1]
        d_jadd2 = serial_data[j + 2]
        grad_value = (d_jminus2-8*d_jminus1+8*d_jadd1-d_jadd2)/(12*t_step)
        data2grad[i, 0] = grad_value

    if (is_torch==False) and to_torch:
        data2grad = torch.from_numpy(data2grad)

    if to_cuda:
        data2grad = data2grad.cuda(device='cuda:' + str(gpu_no))

    return data2grad


# 根据数据序列, 估算每个点处的梯度
def estimate_grad2given_serial_data_date(serial_date, serial_data, size2data=100, t_step=1.0, is_torch=False,
                                         type2float='float32', to_torch=False, to_cuda=False, gpu_no=0,
                                         use_grad2x=False):
    if is_torch:
        if type2float == 'float32':
            float_type = torch.float32
        elif type2float == 'float64':
            float_type = torch.float64
        elif type2float == 'float16':
            float_type = torch.float16
    else:
        if type2float == 'float32':
            float_type = np.float32
        elif type2float == 'float64':
            float_type = np.float64
        elif type2float == 'float16':
            float_type = np.float16

    if is_torch:
        date2grad = torch.empty([size2data-4, 1], dtype=float_type)
        data2grad = torch.empty([size2data - 4, 1], dtype=float_type)
    else:
        date2grad = np.empty([size2data - 4, 1], dtype=float_type)
        data2grad = np.empty([size2data - 4, 1], dtype=float_type)

    for i in range(size2data-4):
        j = i+2
        d_jminus2 = serial_data[j - 2]
        d_jminus1 = serial_data[j - 1]
        dj = serial_data[j - 1]
        d_jadd1 = serial_data[j + 1]
        d_jadd2 = serial_data[j + 2]
        grad_value = (d_jminus2-8*d_jminus1+8*d_jadd1-d_jadd2)/(12*t_step)
        data2grad[i, 0] = grad_value
        date2grad[i, 0] = serial_date[j]

    if (is_torch == False) and to_torch:
        data2grad = torch.from_numpy(data2grad)
        date2grad = torch.from_numpy(date2grad)

        date2grad.requires_grad = use_grad2x

    if to_cuda:
        data2grad = data2grad.cuda(device='cuda:' + str(gpu_no))
        date2grad = date2grad.cuda(device='cuda:' + str(gpu_no))

    return date2grad, data2grad


if __name__ == '__main__':
    trainSet_szie = 280
    batchsize2test = 20

    R = {}

    R['total_population'] = 3450000  # 总的“人口”数量

    # R['normalize_population'] = 3450000                # 归一化时使用的“人口”数值
    # R['normalize_population'] = 50000
    # R['normalize_population'] = 10000
    # R['normalize_population'] = 5000
    # R['normalize_population'] = 2000
    R['normalize_population'] = 1000
    # R['normalize_population'] = 1

    R['use_gpu'] = True

    R['gpuNo'] = 0

    # filename = 'data2csv/Wuhan.csv'
    # filename = 'data2csv/Italia_data.csv'
    # filename = 'data2csv/Korea_data.csv'
    # filename = 'data2csv/minnesota.csv'
    # filename = 'data2csv/minnesota2.csv'
    filename = 'data/minnesota3.csv'
    # 根据文件读入数据，然后存放在 numpy 数组里面
    date, data2S, data2I, data2R, data2D = load_4csvData_cal_S(
        datafile=filename, total_population=R['total_population'])

    assert (trainSet_szie + batchsize2test <= len(data2I))
    if R['normalize_population'] == 1:
        # 不归一化数据
        train_date, train_data2s, train_data2i, train_data2r, train_data2d, test_date, test_data2s, test_data2i, \
        test_data2r, test_data2d = split_5csvData2train_test(
            date, data2S, data2I, data2R, data2D, size2train=trainSet_szie, normalFactor=1.0, to_torch=True,
            to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'], use_grad2x=True)
    elif (R['total_population'] != R['normalize_population']) and R['normalize_population'] != 1:
        # 归一化数据，使用的归一化数值小于总“人口”
        train_date, train_data2s, train_data2i, train_data2r, train_data2d, test_date, test_data2s, test_data2i, \
        test_data2r, test_data2d = split_5csvData2train_test(
            date, data2S, data2I, data2R, data2D, size2train=trainSet_szie, normalFactor=R['normalize_population'],
            to_torch=True, to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'], use_grad2x=True)
    elif (R['total_population'] == R['normalize_population']) and R['normalize_population'] != 1:
        # 归一化数据，使用总“人口”归一化数据
        train_date, train_data2s, train_data2i, train_data2r, train_data2d, test_date, test_data2s, test_data2i, \
        test_data2r, test_data2d = split_5csvData2train_test(
            date, data2S, data2I, data2R, data2D, size2train=trainSet_szie, normalFactor=R['total_population'],
            to_torch=True, to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'], use_grad2x=True)

    # 根据训练数据序列, 估算每个点处的梯度

    date2grad, grad_data2S = estimate_grad2given_serial_data_date(train_date, train_data2s, size2data=trainSet_szie,
                                                                  t_step=1.0, is_torch=True, type2float='float32',
                                                                  to_torch=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'],
                                                                  use_grad2x=False)

    grad_data2I = estimate_grad2given_serial_data(train_data2i, size2data=trainSet_szie, t_step=1.0, is_torch=True,
                                                  type2float='float32', to_torch=True, to_cuda=R['use_gpu'],
                                                  gpu_no=R['gpuNo'])

    grad_data2R = estimate_grad2given_serial_data(train_data2r, size2data=trainSet_szie, t_step=1.0, is_torch=True,
                                                  type2float='float32', to_torch=True, to_cuda=R['use_gpu'],
                                                  gpu_no=R['gpuNo'])

    grad_data2D = estimate_grad2given_serial_data(train_data2d, size2data=trainSet_szie, t_step=1.0, is_torch=True,
                                                  type2float='float32', to_torch=True, to_cuda=R['use_gpu'],
                                                  gpu_no=R['gpuNo'])

    print('end!!!!!')
