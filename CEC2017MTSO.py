"""CEC2017 MTSO 测试集（优化版，接口与原 CEC2017MTSO.py 兼容）。

用法与原文件完全一致：
    task1 = Tasks('CIHS', 1)
    fitness = task1.function(array)      # array: (d,) 或 (1,d)

相对原实现的改动（均为等价变换，详见 函数重写说明.md）：
  1. Weierstrass 的双重 Python 循环改为 (21, d) 广播，常数项预计算；
  2. 一维化入口 _flat()，不再依赖 GO_TaskN 的 (1,d) 存储形状；
  3. Griewank 的 sqrt(1..d) 与 Weierstrass 的 a^k/b^k 提到模块级；
  4. 9 个任务函数收敛为 _TASKS 表 + 一个 _evaluate 分派器（旧函数名保留为 shim）；
  5. get_data 加 lru_cache、绝对路径回退、未知 question 显式报错；
  6. 任务号非法时显式 ValueError，而不是静默返回 None；
  7. 维度优先由 Rotation 矩阵推出，先切片再缩放。
"""

from __future__ import annotations

import functools
import os

import numpy as np
import scipy.io as sio

# ---------------------------------------------------------------------------
# 模块级常量：与输入无关，只算一次
# ---------------------------------------------------------------------------
_KMAX = 20
_AK = 0.5 ** np.arange(_KMAX + 1)                      # a^k,  k = 0..20
_BK = 3.0 ** np.arange(_KMAX + 1)                      # b^k,  k = 0..20
_WCONST = float(np.sum(_AK * np.cos(np.pi * _BK)))     # Σ a^k·cos(2π·0.5·b^k)

_SCHWEFEL_C = 418.9829                                 # 与原实现保持一致


@functools.lru_cache(maxsize=None)
def _griewank_divisor(d):
    """sqrt(1..d)，按维度缓存。"""
    return np.sqrt(np.arange(1, d + 1))


def _flat(x):
    """统一入口形状：(1,d) / (d,) / list 一律视为一维 float64。"""
    return np.asarray(x, dtype=float).ravel()


# ---------------------------------------------------------------------------
# 基本函数
# ---------------------------------------------------------------------------
def Sphere(array, opt):
    # 原：np.sum((a - opt) ** 2)，要 1 次减法 + 1 次平方 + 1 次归约（3 个临时数组）
    # 现：np.dot 走 BLAS，只 1 个临时数组
    z = _flat(array) - _flat(opt)
    return float(np.dot(z, z))


def Rosenbrock(array):
    # 原：array[1:] 与 array[:-1] 各取两次，[:-1] ** 2 生成两次中间数组
    # 现：切片只取一次，语义不变
    a = _flat(array)
    x0 = a[:-1]
    x1 = a[1:]
    return float(np.sum(100.0 * (x1 - x0 * x0) ** 2 + (1.0 - x0) ** 2))


def Ackley(array, M, opt):
    # 原：多出 d = len(array)（未使用）、二维路径、array.T（恒等）
    z = np.dot(M, _flat(array) - _flat(opt))
    s = z * z                                   # 平方只算一次
    res = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(s)))
    res = res - np.exp(np.mean(np.cos(2.0 * np.pi * z))) + 20.0 + np.exp(1.0)
    return float(res)


def Rastrgin(array, M, opt):
    z = np.dot(M, _flat(array) - _flat(opt))
    return float(10.0 * z.size + np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z)))


# 修正拼写的新名字，旧名保留
Rastrigin = Rastrgin


def Griewank(array, M, opt):
    # 原：每次调用 arange + sqrt；现：按维度缓存
    z = np.dot(M, _flat(array) - _flat(opt))
    return float(1.0 + np.sum(z * z) / 4000.0
                 - np.prod(np.cos(z / _griewank_divisor(z.size))))


def Weierstrass(array, M, opt):
    # 原：for i in range(d): for k in range(21): math.cos(...) —— 1050 次标量调用
    # 现：一次 (21, d) 广播；常数项 Σa^k·cos(πb^k) 预先算好
    z = np.dot(M, _flat(array) - _flat(opt))
    ph = 2.0 * np.pi * (_BK[:, None] * (z[None, :] + 0.5))
    return float(np.sum(_AK[:, None] * np.cos(ph)) - z.size * _WCONST)


def Schwefel(array):
    a = _flat(array)
    return float(_SCHWEFEL_C * a.size - np.sum(a * np.sin(np.sqrt(np.abs(a)))))


# ---------------------------------------------------------------------------
# 任务表：把原 9 个函数 × 2 分支压成一张表
#   (基本函数, low, high, 旋转矩阵键, 最优值键, 维度)
#   维度 None 表示由旋转矩阵阶数推出；无旋转的任务必须显式给出
# ---------------------------------------------------------------------------
_R1, _G1, _R2, _G2 = 'Rotation_Task1', 'GO_Task1', 'Rotation_Task2', 'GO_Task2'

_TASKS = {
    # 完全交叉
    'CIHS': {1: (Griewank, -100.0, 100.0, _R1, _G1, None),
             2: (Rastrgin, -50.0, 50.0, _R2, _G2, None)},
    'CIMS': {1: (Ackley, -50.0, 50.0, _R1, _G1, None),
             2: (Rastrgin, -50.0, 50.0, _R2, _G2, None)},
    'CILS': {1: (Ackley, -50.0, 50.0, _R1, _G1, None),
             2: (Schwefel, -500.0, 500.0, None, None, 50)},
    # 部分交叉
    'PIHS': {1: (Rastrgin, -50.0, 50.0, _R1, _G1, None),
             2: (Sphere, -100.0, 100.0, None, _G2, 50)},
    'PIMS': {1: (Ackley, -50.0, 50.0, _R1, _G1, None),
             2: (Rosenbrock, -50.0, 50.0, None, None, 50)},
    'PILS': {1: (Ackley, -50.0, 50.0, _R1, _G1, None),
             2: (Weierstrass, -0.5, 0.5, _R2, _G2, None)},   # 25 维，由 M 推出
    # 无交叉
    'NIHS': {1: (Rosenbrock, -50.0, 50.0, None, None, 50),
             2: (Rastrgin, -50.0, 50.0, _R2, _G2, None)},
    'NIMS': {1: (Griewank, -100.0, 100.0, _R1, _G1, None),
             2: (Weierstrass, -0.5, 0.5, _R2, _G2, None)},
    'NILS': {1: (Rastrgin, -50.0, 50.0, _R1, _G1, None),
             2: (Schwefel, -500.0, 500.0, None, None, 50)},
}


def _evaluate(question, task, array, data):
    """所有任务共用的唯一执行路径。"""
    try:
        fn, low, high, rot_key, opt_key, dim = _TASKS[question][task]
    except KeyError:
        raise ValueError(
            f'未知任务 question={question!r} task={task!r}；'
            f'question 应为 {sorted(_TASKS)}，task 应为 1 或 2') from None

    M = data[rot_key] if rot_key is not None else None
    d = M.shape[0] if M is not None else dim

    # 维度契约：输入可以比任务维度长（多种群统一用 maxD 维向量），只取前 d 维；
    # 但比 d 短就无法定义，显式报错，避免落到 np.dot 里报 "shapes not aligned"。
    x_full = _flat(array)
    if x_full.size < d:
        raise ValueError(
            f'{question} 的 task{task} 需要前 {d} 维，输入只有 {x_full.size} 维')

    # 先切片再缩放：原实现先对 len(array) 维做 l + a*(u-l) 再截断，多算的部分被丢弃
    x = low + (high - low) * x_full[:d]

    if M is not None:
        opt = _flat(data[opt_key])
        return fn(x, M, opt)
    if opt_key is not None:
        return fn(x, _flat(data[opt_key]))
    return fn(x)


# --- 旧接口 shim（保留以便外部脚本直接调用，内部不再使用）------------------
def CIHS(array, task, data):
    return _evaluate('CIHS', task, array, data)


def CIMS(array, task, data):
    return _evaluate('CIMS', task, array, data)


def CILS(array, task, data):
    return _evaluate('CILS', task, array, data)


def PIHS(array, task, data):
    return _evaluate('PIHS', task, array, data)


def PIMS(array, task, data):
    return _evaluate('PIMS', task, array, data)


def PILS(array, task, data):
    return _evaluate('PILS', task, array, data)


def NIHS(array, task, data):
    return _evaluate('NIHS', task, array, data)


def NIMS(array, task, data):
    return _evaluate('NIMS', task, array, data)


def NILS(array, task, data):
    return _evaluate('NILS', task, array, data)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
_DATA_SUBDIR = os.path.join('Questions', 'CEC17')
_FILES = {
    'CIHS': 'CI_H.mat', 'CIMS': 'CI_M.mat', 'CILS': 'CI_L.mat',
    'PIHS': 'PI_H.mat', 'PIMS': 'PI_M.mat', 'PILS': 'PI_L.mat',
    'NIHS': 'NI_H.mat', 'NIMS': 'NI_M.mat', 'NILS': 'NI_L.mat',
}


def _resolve_data_dir():
    """依次尝试环境变量、CWD、模块同级、仓库内位置。"""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get('CEC17_DATA_DIR'), os.path.join('.', _DATA_SUBDIR)]
    cands += [os.path.join(here, _DATA_SUBDIR),
              os.path.join(here, '..', 'test', 'XXMTO', _DATA_SUBDIR),
              os.path.join(here, '..', '..', 'test', 'XXMTO', _DATA_SUBDIR)]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, _FILES['CIHS'])):
            return c
    raise FileNotFoundError(f'找不到 CEC17 数据目录，已尝试：{cands}')


@functools.lru_cache(maxsize=None)
def _load(question):
    if question not in _FILES:
        raise KeyError(f'未知 question={question!r}；应为 {sorted(_FILES)}')
    path = os.path.join(_resolve_data_dir(), _FILES[question])
    raw = sio.loadmat(path)
    # 一次性把 GO 转成一维 float，避免每次评估都做 uint8 广播/类型提升
    return {k: (v.ravel().astype(float) if k.startswith('GO_') else v)
            for k, v in raw.items() if not k.startswith('__')}


def get_data(question):
    return _load(question)


# ---------------------------------------------------------------------------
# 任务封装
# ---------------------------------------------------------------------------
class Tasks:
    def __init__(self, question, task):
        self.question = question
        self.task = task
        self.data = _load(question)
        # 提前校验一次，避免非法 task 在算法深处以 TypeError 形式爆炸
        # （用 100 维探针，比任何任务的维度都长，走的是"只取前 d 维"这条正常路径）
        _evaluate(question, task, np.zeros(100), self.data)
        self._fn = functools.partial(_evaluate, question, task)

    def function(self, array):
        return self._fn(array, self.data)
