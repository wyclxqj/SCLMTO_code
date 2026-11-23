import algorithm
import pandas as pd
import numpy as np
import Benchmark
def run(benchmark,question,res,name,ktp,slp,nt):
    #获取测试集
    Tasks,maxD,filename,T = Benchmark.get_benchmark(benchmark,question,name)
    T1=[[] for i in range(T)]
    # 重复运行
    for i in range(res):
        #运行算法
        gbest_F = algorithm.algorithm(Tasks, question,maxD,ktp,slp,nt)
        arr = np.array(gbest_F)
        #找出均匀的11代以画图
        n_rows, n_cols = arr.shape
        # 选择11个均匀分布的列索引（包括第0列和最后一列）
        # 在0到n_cols-1之间选择11个等间距的点
        indices = np.linspace(0, n_cols - 1, 11, dtype=int)
        # 确保包含首尾（由于浮点运算可能不精确）
        indices[0] = 0
        indices[-1] = n_cols - 1
        # 选择对应的列
        gbest_F = arr[:, indices]
        #分散到每个任务里面
        for t in range(T):
            T1[t].append(gbest_F[t])
    # 将数据写入两个不同的Excel表格
    with pd.ExcelWriter(filename) as writer:
        for t in range(T):
            df = pd.DataFrame(T1[t])
            df.to_excel(writer, sheet_name='Task'+str(t+1), index=False)

#各种测试集
CEC17 = ["CIHS", "CIMS", "CILS", "PIHS", "PIMS", "PILS", "NIHS", "NIMS", "NILS"]
CEC22 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MaTO19=[0,1, 2, 3, 4, 5]
MaTO22=[0,1, 2, 3, 4, 5, 6, 7, 8, 9]

res=30  # 重复运行次数

#本地跑代码
#  运行CEC17
# for i in range(len(CEC17)):
#     run('CEC17',CEC17[i],res,'SCLMTO',ktp=0.1,slp=0.5,nt=3)
#  运行CEC22
# for i in range(len(CEC22)):
#     run('CEC22',CEC22[i],res,'SCLMTO',ktp=0.1,slp=0.5,nt=3)
# #  运行MaTO19
# for i in range(len(MaTO19)):
#     run('MaTO19',MaTO19[i],res,'SCLMTO',ktp=0.1,slp=0.5,nt=3)
# #  运行MaTO22
# for i in range(len(MaTO22)):
#     run('MaTO22',MaTO22[i],res,'SCLMTO',ktp=0.1,slp=0.5,nt=3)




# #服务器跑代码-本身-17 22
# import argparse
# # 创建解析器
# parser = argparse.ArgumentParser(description="Run specific function based on function number")
# parser.add_argument('--func', type=int, required=True, help='Function number')
# # 解析参数
# args = parser.parse_args()
# para=0.1
# if args.func<9:
#     question=CEC17[args.func]
#     run('CEC17',question,res,'SCLMTO',ktp=0.1,slp=0.5,nt=3)
# else:
#     question=CEC22[args.func-9]
#     run('CEC22',question,res,'SCLMTO',ktp=0.1,slp=0.5,nt=3)


# #服务器跑代码-本身-many 19 22
# import argparse
# # 创建解析器
# parser = argparse.ArgumentParser(description="Run specific function based on function number")
# parser.add_argument('--func', type=int, required=True, help='Function number')
# # 解析参数
# args = parser.parse_args()
# para=0.1
# if args.func<6:
#     question=MaTO19[args.func]
#     run('MaTO19',question,res,'SCLMTO',ktp=0.1,slp=0.5,nt=3)
# else:
#     question=MaTO22[args.func-6]
#     run('MaTO22',question,res,'SCLMTO',ktp=0.1,slp=0.5,nt=3)





# #服务器跑代码-灵敏度分析
import argparse
# 创建解析器
parser = argparse.ArgumentParser(description="Run specific function based on function number")
parser.add_argument('--func', type=int, required=True, help='Function number')
parser.add_argument('--name', type=str, required=True, help='Parameter name')
parser.add_argument('--para', type=int, required=True, help='Parameter value')
# 解析参数
args = parser.parse_args()
name=args.name+'='+str(args.para/10)
para=args.para/10
if args.func<9:
    question=CEC17[args.func]
    run('CEC17',question,res,name,para,slp=0.5,nt=3)
else:
    question=CEC22[args.func-9]
    run('CEC22',question,res,name,para,slp=0.5,nt=3)
