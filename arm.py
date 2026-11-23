import scipy.io as sio
import numpy as np
import math
import algorithm as alg
import pandas as pd


class Arm:
    def __init__(self, lengths):
        self.n_dofs = len(lengths)
        self.lengths = np.concatenate(([0], lengths))
        self.joint_xy = []

    def fw_kinematics(self, p):
        from math import cos, sin, pi, sqrt
        assert (len(p) == self.n_dofs)
        p = np.append(p, 0)
        self.joint_xy = []
        mat = np.matrix(np.identity(4))
        for i in range(0, self.n_dofs + 1):
            m = [[cos(p[i]), -sin(p[i]), 0, self.lengths[i]],
                 [sin(p[i]), cos(p[i]), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]
            mat = mat * np.matrix(m)
            v = mat * np.matrix([0, 0, 0, 1]).transpose()
            self.joint_xy += [v[0:2].A.flatten()]
        return self.joint_xy[self.n_dofs], self.joint_xy


class Task:
    def __init__(self, task):
        self.task = task

    def function(self, angles):
        angular_range = self.task[0] / len(angles)
        lengths = np.ones(len(angles)) * self.task[1] / len(angles)
        target = 0.5 * np.ones(2)
        a = Arm(lengths)
        command = (angles - 0.5) * angular_range * math.pi * 2
        ef, _ = a.fw_kinematics(command)
        f = np.sum((ef - target) ** 2)**0.5
        return f


questions = [20, 30, 50, 100, 200, 300, 500]
for question in questions:
    res=30
    maxD=question
    path = './results/'
    # 遍历问题
    Means = []
    Stds = []
    Mins = []
    data = sio.loadmat("./Questions/ARM/" + str(question) + ".mat")
    data = data['task_para']
    print(data)
    task1 = Task(data[0])
    task2 = Task(data[1])
    T1 = []
    T2 = []
    # 重复运行
    for i in range(res):
        t1, t2, t1_gbest_F, t2_gbest_F = alg.algorithm(task1, task2, question,maxD,maxD)
        T1.append(t1)
        T2.append(t2)
    T1 = np.array(T1)
    T2 = np.array(T2)
    # 求最后一列的平均值
    last_column_mean_task1 = np.mean(T1[:, -1])
    last_column_mean_task2 = np.mean(T2[:, -1])
    # 求最后一列的标准差
    last_column_std_task1 = np.std(T1[:, -1])
    last_column_std_task2 = np.std(T2[:, -1])
    # 求最后一列的最小值
    last_column_min_task1 = np.min(T1[:, -1])
    last_column_min_task2 = np.min(T2[:, -1])
    Means.append([last_column_mean_task1, last_column_mean_task2])
    Stds.append([last_column_std_task1, last_column_std_task2])
    Mins.append([last_column_min_task1, last_column_min_task2])
    df1 = pd.DataFrame(T1)
    df2 = pd.DataFrame(T2)
    # 将数据写入两个不同的Excel表格
    with pd.ExcelWriter(path + str(question) + '.xlsx') as writer:
        df1.to_excel(writer, sheet_name='Task1', index=False)
        df2.to_excel(writer, sheet_name='Task2', index=False)
