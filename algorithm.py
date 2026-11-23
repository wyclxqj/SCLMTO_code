import random
import numpy as np
import Population_Operator as po
import operator
import SCL
from scipy.spatial.distance import cdist
G = 1000  # 迭代次数
NP = 50  # 个体数目



# 算法本身
def algorithm(Tasks,question,maxD,ktp=0.1,slp=0.5,nt=3):
    T=len(maxD)
    MaxFEs = 50000 * T
    # 初始化种群
    Pop = po.t_init(maxD,NP,T)
    # 对种群个体进行初始评价
    Pop, gbest_F, gbest_X=po.t_evaluate(Pop,Tasks,maxD,NP,T)
    # 根据因子代价排序
    for t in range(T):
        Pop[t] = sorted(Pop[t], key=operator.itemgetter('cost'))
    #记录每代最优
    Graph=[]
    for t in range(T):
        graph=[]
        Graph.append(graph)
    fes = 0
    g = 0
    # F和Cr参数设置
    para_F_and_Cr = []
    for t in range(T):
        if random.uniform(0,1) < 0.5:
            F=0.9
            Cr=0.2
        else:
            F=0.4
            Cr=0.9
        para_F_and_Cr.append([F,Cr])
    # 进行迭代
    while fes < MaxFEs:

        #遍历每一个任务
        for t in range(T):
            #随机选择一个任务作为源任务
            indexs=[index for index in range(T) if index != t]
            number_of_s=np.random.choice(indexs, 1, replace=False)
            number_of_s= number_of_s[0]

            # 提取种群信息
            pop1 = []
            pop2 = []
            for cnt in range(NP):
                pop1.append(Pop[t][cnt]['information'].copy())
                pop2.append(Pop[number_of_s][cnt]['information'].copy())
            pop1 = np.array(pop1)
            pop2 = np.array(pop2)

            # 生成方案scheme 只包括另一个种群
            scheme1 = np.random.randint(low=0, high=maxD[number_of_s], size=[maxD[t]])
            # 执行相似，按照方案生成临时种群
            scheme1_pop1 = np.empty((NP, 0))
            for cnt in scheme1:
                index_column = pop2[:, cnt].reshape(-1, 1)
                scheme1_pop1 = np.hstack((scheme1_pop1, index_column))
            #执行相关
            relate1 = []
            for k in range(maxD[t]):
                cnt = scheme1[k]
                tmp11 = pop1[:, k].copy()
                tmp22 = pop2[:, cnt].copy()
                pear = SCL.pearson(tmp11, tmp22)
                relate1.append(pear)


            #遍历种群
            for i in range(NP):
                # 以ktp的概率知识迁移
                if random.uniform(0, 1) < ktp:
                    #以slp的概率相似学习
                    if random.uniform(0,1)<slp:
                        # 对于Pop[t][i]['information'],从scheme1_pop1中找到欧氏距离最接近的3个个体。返回索引
                        indexs = []
                        for j in range(NP):
                            indexs.append(j)
                        indexs = np.array(indexs)
                        # 计算欧氏距离
                        distance = []
                        for j in indexs:
                            distance.append(np.linalg.norm(Pop[t][i]['information'] - scheme1_pop1[j]))
                        distance = np.array(distance)
                        # 排序
                        indexs = indexs[np.argsort(distance)]
                        # 找到最近的十个个体
                        indexs = indexs[:nt]
                        # indexs = [index for index in range(NP) if index != i]
                        r1, r2, r3 = np.random.choice(indexs, 3, replace=False)  # 选3个不相同的下标
                        # 计算平均距离
                        mean_d = (distance[r1] + distance[r2] + distance[r3]) / 3
                        s = 1 / (1 + mean_d)
                        d = SCL.t_DE(scheme1_pop1[r1], scheme1_pop1[r2], scheme1_pop1[r3], Pop[t][i], maxD[t], 0.5,
                                     s)
                    #以1-slp的概率相关学习
                    else:
                        # 两个种群选择一个个体
                        indexs2 = [index for index in range(NP) if Pop[number_of_s][index]['better'] == True]
                        r2 = np.random.choice(indexs2)
                        tmp = []
                        for cnt in range(maxD[t]):
                            tmp.append(Pop[number_of_s][r2]['updown'][scheme1[cnt]])
                        d = SCL.local_search(Pop[t][i]['information'], maxD[t], tmp, relate1)
                # 以1-ktp的概率自我进化
                else:
                    indexs = [index for index in range(NP) if index != i]
                    r1, r2, r3 = np.random.choice(indexs, 3, replace=False)  # 选3个不相同的下标
                    d = po.t_DE(Pop[t][i], Pop[t][r1], Pop[t][r2], Pop[t][r3], para_F_and_Cr[t][0],
                                para_F_and_Cr[t][1], maxD[t])

                d['cost'] = Tasks[t].function(d["information"])
                Pop[t].append(d)
                fes += 1
                #记录进化方向
                if d['cost'] < Pop[t][i]['cost']:
                    d['better'] = True
                    d['updown'] = d['information'].copy() - Pop[t][i]['information'].copy()
                    Pop[t][i]['better'] = True
                    Pop[t][i]['updown'] = d['updown'].copy()
                else:
                    d['better'] = True
                    d['updown'] = Pop[t][i]['information'].copy() - d['information'].copy()
                    Pop[t][i]['better'] = True
                    Pop[t][i]['updown'] = d['updown'].copy()
        # 当代个体生成完毕，且适应值求解完成，开始选择操作
        # 根据因子代价排序
        for t in range(T):
            Pop[t] = sorted(Pop[t], key=operator.itemgetter('cost'))
            # 删除后面部分,2NP个个体只保留前NP个个体
            for i in range(NP):
                Pop[t].pop()

        # 更新最优
        for t in range(T):
            Tag = False
            for i in range(NP):
                if gbest_F[t] > Pop[t][i]["cost"]:  # 更新任务1最优
                    gbest_F[t] = Pop[t][i]["cost"]
                    gbest_X[t] = Pop[t][i]["information"].copy()
                    Tag = True
            if Tag == False:
                if para_F_and_Cr[t][0] ==0.9:
                    para_F_and_Cr[t][0] = 0.4
                    para_F_and_Cr[t][1] = 0.9
                else:
                    para_F_and_Cr[t][0] = 0.9
                    para_F_and_Cr[t][1] = 0.2

        # 记录当前代最优
        for t in range(T):
            Graph[t].append(gbest_F[t])
        g += 1
        if g % 50 == 0:
            print(g, gbest_F[0], gbest_F[1])  # 输出当前代两个任务的最优值
    print(question, gbest_F[0], gbest_F[1])  # 输出question问题下两个任务的最优值

    return Graph
