import torch
import numpy as np
import matplotlib.pyplot as plt


# RK4----------------------------------------------
# 定义方程
def dsdt_func(t=None, s=None, i=None, r=None, d=None, beta=None, gamma=None, mu=None):
    ds_dt = - beta * s * i / (s + i)
    return ds_dt


def didt_func(t=None, s=None, i=None, r=None, d=None, beta=None, gamma=None, mu=None):
    di_dt = beta * s * i/(s + i) - gamma * i - mu * i
    return di_dt


def drdt_func(t=None, s=None, i=None, r=None, d=None, beta=None, gamma=None, mu=None):
    dr_dt = gamma * i
    return dr_dt


def dddt_func(t=None, s=None, i=None, r=None, d=None, beta=None, gamma=None, mu=None):
    dd_dt = mu * i
    return dd_dt


# 使用Runge-Kutta 更新微分方程的值。输入当前值和当前时刻，以及时间步长，得到下一时刻的值
def SIRD_RK4(t=None, s0=20, i0=10, r0=5, d0=4, h=None, beta=None, gamma=None, mu=None):
    """
    Args:
        t: 时间点
        s0: s的当前值
        i0: i的当前值
        r0: r的当前值
        d0: d的当前值
        h: t的步长
        beta:
        gamma:
        mu:
    Returns:
        迭代更新后的值
    """
    # t += h
    Ks_1 = dsdt_func(t=t, s=s0, i=i0, r=r0, d=d0, beta=beta, gamma=gamma, mu=mu)
    Ki_1 = didt_func(t=t, s=s0, i=i0, r=r0, d=d0, beta=beta, gamma=gamma, mu=mu)
    Kr_1 = drdt_func(t=t, s=s0, i=i0, r=r0, d=d0, beta=beta, gamma=gamma, mu=mu)
    Kd_1 = dddt_func(t=t, s=s0, i=i0, r=r0, d=d0, beta=beta, gamma=gamma, mu=mu)

    Ks_2 = dsdt_func(t=t+h/2, s=s0+h/2*Ks_1, i=i0+h/2*Ks_1, r=r0+h/2*Ks_1, d=d0+h/2*Ks_1, beta=beta, gamma=gamma, mu=mu)
    Ki_2 = didt_func(t=t+h/2, s=s0+h/2*Ki_1, i=i0+h/2*Ki_1, r=r0+h/2*Ki_1, d=d0+h/2*Ki_1, beta=beta, gamma=gamma, mu=mu)
    Kr_2 = drdt_func(t=t+h/2, s=s0+h/2*Kr_1, i=i0+h/2*Kr_1, r=r0+h/2*Kr_1, d=d0+h/2*Kr_1, beta=beta, gamma=gamma, mu=mu)
    Kd_2 = dddt_func(t=t+h/2, s=s0+h/2*Kd_1, i=i0+h/2*Kd_1, r=r0+h/2*Kd_1, d=d0+h/2*Kd_1, beta=beta, gamma=gamma, mu=mu)

    Ks_3 = dsdt_func(t=t+h/2, s=s0+h/2*Ks_2, i=i0+h/2*Ks_2, r=r0+h/2*Ks_2, d=d0+h/2*Ks_2, beta=beta, gamma=gamma, mu=mu)
    Ki_3 = didt_func(t=t+h/2, s=s0+h/2*Ki_2, i=i0+h/2*Ki_2, r=r0+h/2*Ki_2, d=d0+h/2*Ki_2, beta=beta, gamma=gamma, mu=mu)
    Kr_3 = drdt_func(t=t+h/2, s=s0+h/2*Kr_2, i=i0+h/2*Kr_2, r=r0+h/2*Kr_2, d=d0+h/2*Kr_2, beta=beta, gamma=gamma, mu=mu)
    Kd_3 = dddt_func(t=t+h/2, s=s0+h/2*Kd_2, i=i0+h/2*Kd_2, r=r0+h/2*Kd_2, d=d0+h/2*Kd_2, beta=beta, gamma=gamma, mu=mu)

    Ks_4 = dsdt_func(t=t+h, s=s0+h*Ks_3, i=i0+h*Ks_3, r=r0+h*Ks_3, d=d0+h*Ks_3, beta=beta, gamma=gamma, mu=mu)
    Ki_4 = didt_func(t=t+h, s=s0+h*Ki_3, i=i0+h*Ki_3, r=r0+h*Ki_3, d=d0+h*Ki_3, beta=beta, gamma=gamma, mu=mu)
    Kr_4 = drdt_func(t=t+h, s=s0+h*Kr_3, i=i0+h*Kr_3, r=r0+h*Kr_3, d=d0+h*Kr_3, beta=beta, gamma=gamma, mu=mu)
    Kd_4 = dddt_func(t=t+h, s=s0+h*Kd_3, i=i0+h*Kd_3, r=r0+h*Kd_3, d=d0+h*Kd_3, beta=beta, gamma=gamma, mu=mu)

    s = s0 + (Ks_1 + 2 * Ks_2 + 2 * Ks_3 + Ks_4) * h / 6
    i = i0 + (Ki_1 + 2 * Ki_2 + 2 * Ki_3 + Ki_4) * h / 6
    r = r0 + (Kr_1 + 2 * Kr_2 + 2 * Kr_3 + Kr_4) * h / 6
    d = d0 + (Kd_1 + 2 * Kd_2 + 2 * Kd_3 + Kd_4) * h / 6

    return s, i, r, d


def test_RK4_SIRD():
    n = 100
    t_temp = np.arange(1, n)
    t_arr = np.concatenate([[0], t_temp], axis=-1)
    s_init = 1000
    i_init = 500
    r_init = 2
    d_init = 1
    s_list = []
    i_list = []
    r_list = []
    d_list = []
    beta = 0.075
    gamma = 0.01
    mu = 0.0025
    # s_list.append(s_init)
    # i_list.append(i_init)
    # r_list.append(r_init)
    # d_list.append(d_list)
    for i in range(n):
        # temp = t_arr[i]
        s, i, r, d = SIRD_RK4(t=t_arr[i], s0=s_init, i0=i_init, r0=r_init, d0=d_init,
                              h=1.0, beta=beta, gamma=gamma, mu=mu)
        s_init = s
        i_init = i
        r_init = r
        d_init = d
        s_list.append(s)
        i_list.append(i)
        r_list.append(r)
        d_list.append(d)

    print('sshshsshkda')

    ax = plt.gca()
    ax.plot(t_arr, s_list, 'b-.', label='s')
    ax.plot(t_arr, i_list, 'r-*', label='i')
    ax.plot(t_arr, r_list, 'k:', label='r')
    ax.plot(t_arr, d_list, 'c--', label='d')
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='right', bbox_to_anchor=(0.9, 1.05), ncol=4, fontsize=12)
    ax.set_xlabel('t', fontsize=14)
    ax.set_ylabel('s', fontsize=14)

    plt.show()


if __name__ == '__main__':
    test_RK4_SIRD()