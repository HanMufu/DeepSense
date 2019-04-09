import random
import numpy as np
from scipy.fftpack import fft

'''交错合并两个list，用来合并傅立叶之后的实部虚部'''
def merge_f(x, y):
    lst = []
    for i in list(zip(x, y)):
        lst.append(list(i))
    m = []
    for i in lst:
        for j in i:
            m.append(j)
    return m


'''faker function'''
'''generate test data'''


def generate_fake_data():
    #     half_T = random.randint(2, 10)
    half_T = 10
    T = half_T * 2

    acc_x_constant = random.randint(1, 10)
    acc_y_constant = random.randint(1, 10)
    acc_x_1 = [acc_x_constant, ] * 6
    acc_y_1 = [acc_y_constant, ] * 6
    # add random noise
    acc_x_1 += np.random.normal(size=6)
    acc_y_1 += np.random.normal(size=6)
    acc_x = [acc_x_1, ] * half_T
    acc_y = [acc_y_1, ] * half_T
    for i in range(0, half_T):
        acc_x.append((np.array(acc_x_1) * -1).tolist())
        acc_y.append((np.array(acc_y_1) * -1).tolist())

    for i in range(0, len(acc_x)):
        acc_x_fft = fft(acc_x[i])
        acc_x_fft_real = acc_x_fft.real
        acc_x_fft_imag = acc_x_fft.imag
        acc_x[i] = merge_f(acc_x_fft_real, acc_x_fft_imag)
    for i in range(0, len(acc_y)):
        acc_y_fft = fft(acc_y[i])
        acc_y_fft_real = acc_y_fft.real
        acc_y_fft_imag = acc_y_fft.imag
        acc_y[i] = merge_f(acc_y_fft_real, acc_y_fft_imag)
    acc = merge_f(acc_x, acc_y)
    acc = [acc[i:i + 2] for i in range(0, len(acc), 2)]
    # padding
    for i in range(T, 20):
        acc.append([[0., ] * 12, [0., ] * 12])
    # 大功告成，得到speed的输入
    acc = np.array(acc)

    '''start to generate speed data'''
    # 这里list和np.array的转换有些多余
    # 但我想用向量点乘，又想用list.append，所以只能转来转去了
    # any better solutions?
    speed_x = (np.array([i for i in range(0, 30 * half_T, 5)]) * np.array([acc_x_constant, ] * (6 * half_T))).tolist()
    speed_x = [speed_x[i:i + 6] for i in range(0, len(speed_x), 6)]
    # 为了最大限度模仿真实数据，我们要让speed数据的第一时刻和最后一时刻等于0
    # 所以这个speed序列模仿的小车移动是匀加速+匀减速直线运动，其中加速和减速运动中的加速度大小相等，方向相反
    speed_x_reversed = sorted(speed_x, reverse=True)
    for i in range(0, len(speed_x_reversed)):
        speed_x_reversed[i] = sorted(speed_x_reversed[i], reverse=True)
    speed_x = speed_x + speed_x_reversed

    speed_y = (np.array([i for i in range(0, 30 * half_T, 5)]) * np.array([acc_y_constant, ] * (6 * half_T))).tolist()
    speed_y = [speed_y[i:i + 6] for i in range(0, len(speed_y), 6)]
    speed_y_reversed = sorted(speed_y, reverse=True)
    for i in range(0, len(speed_y_reversed)):
        speed_y_reversed[i] = sorted(speed_y_reversed[i], reverse=True)
    speed_y = speed_y + speed_y_reversed

    for i in range(0, len(speed_x)):
        speed_x_fft = fft(speed_x[i])
        speed_x_fft_real = speed_x_fft.real
        speed_x_fft_imag = speed_x_fft.imag
        speed_x[i] = merge_f(speed_x_fft_real, speed_x_fft_imag)

    for i in range(0, len(speed_y)):
        speed_y_fft = fft(speed_y[i])
        speed_y_fft_real = speed_y_fft.real
        speed_y_fft_imag = speed_y_fft.imag
        speed_y[i] = merge_f(speed_y_fft_real, speed_y_fft_imag)

    speed = merge_f(speed_x, speed_y)
    speed = [speed[i:i + 2] for i in range(0, len(speed), 2)]
    # padding
    for i in range(T, 20):
        speed.append([[0., ] * 12, [0., ] * 12])
    # 大功告成，得到speed的输入
    speed = np.array(speed)

    '''start to generate label'''
    # 先计算匀加速运动时间内的位移
    # 这里half_T要加1，因为我们要计算的位移是不要第一个数(0)的
    displacement_x = 0.5 * acc_x_constant * np.array([i for i in range(0, 30 * (half_T + 1), 30)]) * np.array(
        [i for i in range(0, 30 * (half_T + 1), 30)])
    displacement_y = 0.5 * acc_y_constant * np.array([i for i in range(0, 30 * (half_T + 1), 30)]) * np.array(
        [i for i in range(0, 30 * (half_T + 1), 30)])
    displacement_x = displacement_x.tolist()
    displacement_y = displacement_y.tolist()
    displacement_x.remove(displacement_x[0])
    displacement_y.remove(displacement_y[0])
    # 然后计算匀减速运动时间内的位移
    v0_x = (30 * half_T) * acc_x_constant
    v0_y = (30 * half_T) * acc_y_constant
    displacement_x_2nd_phase = np.array([v0_x, ] * (half_T + 1)) * np.array(
        [i for i in range(0, 30 * (half_T + 1), 30)]) - 0.5 * acc_x_constant * np.array(
        [i for i in range(0, 30 * (half_T + 1), 30)]) * np.array([i for i in range(0, 30 * (half_T + 1), 30)])
    displacement_y_2nd_phase = np.array([v0_y, ] * (half_T + 1)) * np.array(
        [i for i in range(0, 30 * (half_T + 1), 30)]) - 0.5 * acc_y_constant * np.array(
        [i for i in range(0, 30 * (half_T + 1), 30)]) * np.array([i for i in range(0, 30 * (half_T + 1), 30)])
    # 为了用remove又转成list类型
    displacement_x_2nd_phase = displacement_x_2nd_phase.tolist()
    displacement_y_2nd_phase = displacement_y_2nd_phase.tolist()
    # 用remove去掉不要的值，不要的值是0
    displacement_x_2nd_phase.remove(displacement_x_2nd_phase[0])
    displacement_y_2nd_phase.remove(displacement_y_2nd_phase[0])
    # 然后把匀加速段和匀减速段拼起来
    displacement_x_2nd_phase = np.array(displacement_x_2nd_phase) + np.array([displacement_x[half_T - 1], ] * half_T)
    displacement_x = displacement_x + displacement_x_2nd_phase.tolist()
    displacement_y_2nd_phase = np.array(displacement_y_2nd_phase) + np.array([displacement_y[half_T - 1], ] * half_T)
    displacement_y = displacement_y + displacement_y_2nd_phase.tolist()
    # 把displacement_x和displacement_y交错拼接起来，变成需要的数据格式
    displacement = merge_f(displacement_x, displacement_y)
    displacement = [displacement[i:i + 2] for i in range(0, T * 2, 2)]
    # padding
    for i in range(T, 20):
        displacement.append([0, 0])
    # 转成np.array格式
    displacement = np.array(displacement)

    return acc, speed, displacement


