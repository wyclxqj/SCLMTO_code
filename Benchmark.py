import Questions.CEC22.WCCI2020MTSO as WCCI
import Questions.CEC17.CEC2017MTSO as CEC
import Questions.MaTO19.tasks19 as t19
import Questions.MaTO22.tasks20 as t22
import pandas as pd
import numpy as np
import os
def get_benchmark(benchmark,question,name):
    # 各种测试集
    if benchmark == 'CEC17':
        if question == 'PILS':
            maxD = [50, 25]
        else:
            maxD = [50, 50]
        T = 2
        task1 = CEC.Tasks(question, 1)
        task2 = CEC.Tasks(question, 2)
        Tasks = [task1, task2]
    elif benchmark == 'CEC22':
        pathw = './results/Benchmark'
        maxD = [50, 50]
        T = 2
        if question == 1:
            task1, task2 = WCCI.Benchmark1()
        elif question == 2:
            task1, task2 = WCCI.Benchmark2()
        elif question == 3:
            task1, task2 = WCCI.Benchmark3()
        elif question == 4:
            task1, task2 = WCCI.Benchmark4()
        elif question == 5:
            task1, task2 = WCCI.Benchmark5()
        elif question == 6:
            task1, task2 = WCCI.Benchmark6()
        elif question == 7:
            task1, task2 = WCCI.Benchmark7()
        elif question == 8:
            task1, task2 = WCCI.Benchmark8()
        elif question == 9:
            task1, task2 = WCCI.Benchmark9()
        elif question == 10:
            task1, task2 = WCCI.Benchmark10()
        Tasks = [task1, task2]
    elif benchmark == 'MaTO19':
        T = 50
        maxD = [50] * 50
        Tasks = t19.MaTO19(question)
    elif benchmark == 'MaTO22':
        T = 50
        maxD = [50] * 50
        Tasks = t22.WCCI2020MaTO(question)
    elif benchmark == 'ARM':
        maxD = [int(question)] * 2

    pathw = './results/' + name
    if os.path.exists(pathw) and os.path.isdir(pathw):
        pathw = pathw + '/results/'
    else:
        os.makedirs(pathw)
        pathw = pathw + '/results/'
        os.makedirs(pathw)
    if benchmark == 'CEC17':
        filename = pathw + str(question) + '.xlsx'
    elif benchmark == 'CEC22':
        filename = pathw + 'Benchmark' + str(question) + '.xlsx'
    elif benchmark == 'MaTO19':
        filename = pathw + 'MaTO19_' + str(question + 1) + '.xlsx'
    elif benchmark == 'MaTO22':
        filename = pathw + 'MaTO22_' + str(question + 1) + '.xlsx'
    elif benchmark == 'ARM':
        filename = pathw + 'ARM_' + str(question) + '.xlsx'
    return Tasks, maxD, filename,T
