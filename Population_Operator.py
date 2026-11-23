import random
import numpy as np

mu = 10  # SBX交叉的参数
mum = 5  # 多项式变异的参数


# 多种群初始化，随机生成NP个个体，每个个体的维数为所有任务中最大的那个
def t_init(maxD, NP=50, T=2):
    Pop = []
    #遍历每一个任务，生成T个种群
    for t in range(T):
        pop1 = []
        #每隔任务遍历每一个个体，生成NP个个体
        for i in range(NP):
            tmp = {}
            tmp["information"] = np.random.uniform(low=0, high=1, size=[maxD[t]])
            tmp['better'] = True
            tmp['updown'] = np.random.uniform(low=0, high=1, size=[maxD[t]])
            pop1.append(tmp)
        Pop.append(pop1)
    return Pop


# 对于初始多种群population，就task问题，进行初步评价,计算适应值，返回最新种群信息和最优信息
def t_evaluate(Pop, task, maxD, NP=50, T=2):
    gbest_F = [np.inf for t in range(T)]
    gbest_X = [[] for t in range(T)]
    #遍历每一个任务
    for t in range(T):
        population = Pop[t]
        # 对每一个个体评价适应值
        for i in range(NP):
            population[i]["cost"] = task[t].function(population[i]["information"])
            if gbest_F[t] > population[i]["cost"]:  # 更新任务1最优
                gbest_F[t] = population[i]["cost"]
                gbest_X[t] = population[i]["information"].copy()
    return Pop, gbest_F, gbest_X


#SBX 模拟二进制交叉
def t_SBX(population1, population2, i, j, maxD):
    c1 = []  # 新个体1的information
    c2 = []  # 新个体2的information
    u = np.random.uniform(low=0, high=1, size=[maxD])
    cf = np.zeros(maxD)
    for x in range(maxD):
        if u[x] <= 0.5:
            cf[x] = (2 * u[x]) ** (1 / (mu + 1))
        else:
            cf[x] = (2 * (1 - u[x])) ** (-1 / (mu + 1))
        tmp = 0.5 * ((1 - cf[x]) * population1[i]["information"][x] + (1 + cf[x]) * population2[j]["information"][x])
        c1.append(tmp)
        tmp = 0.5 * ((1 + cf[x]) * population1[i]["information"][x] + (1 - cf[x]) * population2[j]["information"][x])
        c2.append(tmp)
    c1 = np.array(c1)  # 列表转nparray
    c2 = np.array(c2)  # 列表转nparray
    # 控边界
    # 将大于1的元素设为1，小于0的元素设为0
    c1[c1 > 1] = 1
    c1[c1 < 0] = 0
    c2[c2 > 1] = 1
    c2[c2 < 0] = 0
    d1 = {}  # 新个体1的整体信息
    d2 = {}  # 新个体2的整体信息
    d1["information"] = c1
    d2["information"] = c2
    return d1, d2


# 多项式变异
def t_PM(array1, maxD):
    c = []
    d = {}
    for x in range(maxD):
        rand = random.uniform(0, 1)
        if rand < 1 / maxD:  # 以1/maxD的概率变异
            u = random.uniform(0, 1)
            if u <= 0.5:  # 以第一个公式变异
                tmp1 = (2 * u) ** (1 / (1 + mum)) - 1
                tmp2 = array1["information"][x] + tmp1 * array1["information"][x]
                c.append(tmp2)
            else:  # 以第二个公式变异
                tmp1 = 1 - (2 * (1 - u)) ** (1 / (1 + mum))
                tmp2 = array1["information"][x] + tmp1 * (1 - array1["information"][x])
                c.append(tmp2)
        else:  # 不变异
            c.append(array1["information"][x])
    c = np.array(c)
    # 控边界
    c[c > 1] = 1
    c[c < 0] = 0
    d["information"] = c
    return d

#DE/rand/1
def t_DE(xi, xr1, xr2, xr3, F, Cr, maxD):
    # 执行变异操作
    v = xr1['information'] + F * (xr2['information'] - xr3['information'])
    # 保证解的有效性
    v[v < 0] = 0
    v[v > 1] = 1
    # 交叉操作
    krand = random.randint(0, maxD - 1)  # jrand
    rand = np.random.uniform(low=0, high=1, size=[maxD])  # rand
    u = xi['information'].copy()
    for k in range(maxD):
        if rand[k] <= Cr or k == krand:
            u[k] = v[k]
    d = {}
    d['information'] = u.copy()
    return d

#DE/best/1
def t_DEbest(xi, xbest, xr1, xr2, F, Cr, maxD):
    # 执行变异操作
    v = xbest['information'] + F * (xr1['information'] - xr2['information'])
    # 保证解的有效性
    v[v < 0] = 0
    v[v > 1] = 1
    # 交叉操作
    krand = random.randint(0, maxD - 1)  # jrand
    rand = np.random.uniform(low=0, high=1, size=[maxD])  # rand
    u = xi['information'].copy()
    for k in range(maxD):
        if rand[k] <= Cr or k == krand:
            u[k] = v[k]
    d = {}
    d['information'] = u.copy()
    return d

#DE/current_to_best/1
def t_DEcurrentTobest(xi, xbest, xr1, xr2, F, Cr, maxD):
    # 执行变异操作
    v = xi['information'] + F * (xbest['information'] - xi['information']) + F * (
                xr1['information'] - xr2['information'])
    # 保证解的有效性
    v[v < 0] = 0
    v[v > 1] = 1
    # 交叉操作
    krand = random.randint(0, maxD - 1)  # jrand
    rand = np.random.uniform(low=0, high=1, size=[maxD])  # rand
    u = xi['information'].copy()
    for k in range(maxD):
        if rand[k] <= Cr or k == krand:
            u[k] = v[k]
    d = {}
    d['information'] = u.copy()
    return d

#高斯搜索
def Gauss_LS(array,fes,MaxFEs,maxD):
    gamma = 1 + fes / MaxFEs * 4
    sigma = 10 ** (-gamma)
    noise= np.random.normal(0, sigma, maxD)
    d = {}
    d['information'] = array['information']+noise
    d['information'] = np.clip(d['information'], 0, 1)
    return d
