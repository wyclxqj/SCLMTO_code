import numpy as np
import random


#计算两组数据的皮尔逊相关系数
def pearson(x, y):
    x = np.array(x)
    y = np.array(y)
    if np.all(x == x[0]):
        x[0] += 1e-5
    if np.all(y == y[0]):
        y[0] += 1e-5
    # 计算均值
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    # 计算协方差和标准差
    covariance = np.sum((x - mean_x) * (y - mean_y))
    std_x = np.sqrt(np.sum((x - mean_x) ** 2))
    std_y = np.sqrt(np.sum((y - mean_y) ** 2))
    # 计算相关系数
    r = covariance / (std_x * std_y)
    return r

def t_DE(array1, array2, array3, array4, maxD, F, CR):
    # 执行变异操作
    v = array1 + F * (array2 - array3)
    # 保证解的有效性
    v[v < 0] = 0
    v[v > 1] = 1
    # 交叉操作
    krand = random.randint(0, maxD - 1)  # jrand
    rand = np.random.uniform(low=0, high=1, size=[maxD])  # rand
    u = array4['information'].copy()
    for k in range(maxD):
        if rand[k] <= CR or k == krand:
            u[k] = v[k]
    d = {}
    d['information'] = u.copy()
    return d
# 基于高斯的局部搜索策略
def local_search(array, maxD, tmp, relate):
    relate=np.array(relate)
    d = {}
    d['information'] = np.random.uniform(low=0, high=1, size=[maxD])
    for j in range(maxD):
        d['information'][j] = array[j] + tmp[j] * relate[j]

    for j in range(maxD):
        if d['information'][j] > 1:
            d['information'][j] = 1
        if d['information'][j] < 0:
            d['information'][j] = 0
    return d