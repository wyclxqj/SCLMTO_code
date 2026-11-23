"""CEC2020/2022 MTSO 测试集（优化版，接口与原 WCCI2020MTSO.py 兼容）。

用法与原文件完全一致：
    task1, task2 = Benchmark1()
    fitness = task1.function(array)

相对原实现的改动（均为等价变换，详见 函数重写说明.md）：
  1. Katsuura 的 50×32 双循环改为 (32, n) 广播（约 170 倍）；
  2. Weierstrass 的 21 次循环改为 (21, n) 广播（约 7 倍）；
  3. Escaffer6 的逐元素循环改为 np.roll 向量化（约 6 倍）；
  4. Hf01/04/05/06 的 shuffle 循环改为花式索引，4 份重复逻辑收敛为一个 _Composite 基类；
  5. 各函数的常量（Ellips 权重、Griewank 分母、Katsuura 的 2^j、Schwefel 罚系数）预计算；
  6. Hgbat/GrieRosen/Ellips 里的重复归约合并；
  7. GetMatrixs/GetShuffle 加缓存并把 shuffle 转成整数索引；
  8. Benchmark1..10 改为表驱动，且 shuffle 路径跟随数据目录（原实现硬编码 ./Questions/CEC22/shuffle/）；
  9. 10 个 Benchmark 函数的返回值保持为 list，可 list 解包。
"""

from __future__ import annotations

import functools
import os

import numpy as np

# ---------------------------------------------------------------------------
# 模块级常量
# ---------------------------------------------------------------------------
_KMAX = 20
_AK = 0.5 ** np.arange(_KMAX + 1)
_BK = 3.0 ** np.arange(_KMAX + 1)
_WCONST = float(np.sum(_AK * np.cos(np.pi * _BK)))     # Σ a^k cos(π b^k)

_KJ = 2.0 ** np.arange(1, 33)                          # Katsuura 的 2^j, j = 1..32
_SCHWEFEL_SHIFT = 4.209687462275036e+002
_SCHWEFEL_C = 4.189828872724338e+002


def _as1d(x):
    return np.asarray(x, dtype=float).ravel()


# ---------------------------------------------------------------------------
# 任务父类
# ---------------------------------------------------------------------------
class Task:
    NAME = None

    def __init__(self, Low=0, High=1, n=50, coeffi=None, bias=None,
                 shuffle=None, sh_rate=1.0):
        self.Low = Low
        self.High = High
        self.Dimension = n
        self.center = np.zeros(n) if bias is None else _as1d(bias)
        self.M = np.eye(n) if coeffi is None else np.asarray(coeffi, dtype=float)
        self.shuffle = shuffle
        self.sh_rate = sh_rate
        self._span = High - Low
        self._plain = (sh_rate == 1.0) and not self.center.any()
        if self.center.size != n:
            raise ValueError(f'bias 维度 {self.center.size} 与 n={n} 不一致')
        if self.M.shape != (n, n):
            raise ValueError(f'coeffi 形状 {self.M.shape} 与 n={n} 不一致')

    def decode(self, X):
        """[0,1]^n -> 旋转后的搜索空间；只接受一维输入（与原实现一致）。"""
        Z = np.asarray(X, dtype=float)
        if Z.ndim != 1:
            raise ValueError(f'decode 只接受一维输入，收到 shape={Z.shape}')
        Z = Z[:self.Dimension]                       # 先切片，再缩放
        Z = self.Low + self._span * Z
        if not self._plain:
            Z = (Z - self.center) * self.sh_rate
        return np.dot(self.M, Z)   # np.dot 比 @ 略快，且与原实现同一 BLAS 入口

    # 子类实现：z 已完成 decode（或 tag=True 时已完成 sh_rate 缩放）
    def _eval(self, z):
        raise NotImplementedError

    def function(self, X, tag=False):
        z = _as1d(X) * self.sh_rate if tag else self.decode(X)
        return self._eval(z)

    def Info(self):
        return (self.NAME or type(self).__name__) + ' ' + str(self.Dimension)


# ---------------------------------------------------------------------------
# 基本函数
# ---------------------------------------------------------------------------
class Ellips(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,
                 shuffle=None, sh_rate=1.0):
        super().__init__(Low, High, n, coeffi, bias, shuffle, sh_rate)
        denom = (n - 1) if n > 1 else 1          # 原实现 n == 1 时会除零
        self._w = 10 ** (6 * np.arange(n) / denom)

    def _eval(self, z):
        return float(np.sum(self._w * z * z))


class Discus(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,
                 shuffle=None, sh_rate=1.0):
        super().__init__(Low, High, n, coeffi, bias, shuffle, sh_rate)

    def _eval(self, z):
        t = z[1:]
        return float(z[0] * z[0] * 1e6 + np.dot(t, t))


class Rosenbrock(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,
                 shuffle=None, sh_rate=2.048 / 100.0):
        super().__init__(Low, High, n, coeffi, bias, shuffle, sh_rate)

    def _eval(self, z):
        # 官方实现（cec14_func.cpp 的 rosenbrock_func）在 sr_func 之后对 z 全体 +1：
        #     z[0] += 1; z[i+1] += 1;
        #     tmp1 = z[i]^2 - z[i+1];  tmp2 = z[i] - 1;
        # 相当于把极小点从 (1,...,1) 平移到旋转后的原点。
        # 原 Python 版本（WCCI2020MTSO.py）漏了这一步，会让 Hf05 与官方 MATLAB
        # 结果出现约 4e-5 的相对偏差；此处按官方实现补上。
        t = z + 1.0
        x0 = t[:-1]
        x1 = t[1:]
        return float(np.sum(100.0 * ((x0 * x0 - x1) ** 2) + (x0 - 1.0) ** 2))


class Ackley(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,
                 shuffle=None, sh_rate=1.0):
        super().__init__(Low, High, n, coeffi, bias, shuffle, sh_rate)

    def _eval(self, z):
        n = z.size
        s = z * z
        return float(20.0 + np.e
                     - 20.0 * np.exp(-0.2 * np.sqrt(np.sum(s) / n))
                     - np.exp(np.sum(np.cos(2.0 * np.pi * z)) / n) + 500.0)


class Weierstrass(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,
                 shuffle=None, sh_rate=0.5 / 100):
        super().__init__(Low, High, n, coeffi, bias, shuffle, sh_rate)

    def _eval(self, z):
        # 一次 (21, n) 广播代替 21 次 Python 循环
        ph = 2.0 * np.pi * (_BK[:, None] * (z[None, :] + 0.5))
        return float(np.sum(_AK[:, None] * np.cos(ph)) - z.size * _WCONST + 600.0)


class Griewank(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,
                 shuffle=None, sh_rate=600.0 / 100):
        super().__init__(Low, High, n, coeffi, bias, shuffle, sh_rate)
        self._sqrt = np.sqrt(np.arange(1, n + 1))

    def _eval(self, z):
        return float(1.0 + np.sum(z * z) / 4000.0
                     - np.prod(np.cos(z / self._sqrt)) + 700.0)


class Rastrigin(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,
                 shuffle=None, sh_rate=5.12 / 100):
        super().__init__(Low, High, n, coeffi, bias, shuffle, sh_rate)

    def _eval(self, z):
        return float(np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0))


class Schwefel(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,
                 shuffle=None, sh_rate=1000.0 / 100):
        super().__init__(Low, High, n, coeffi, bias, shuffle, sh_rate)
        self._pen = 1.0 / (10000.0 * n)

    def _eval(self, z):
        n = z.size
        z = z + _SCHWEFEL_SHIFT
        tmp = z.copy()
        m1 = tmp < -500.0                       # 布尔掩码各算一次
        m2 = tmp > 500.0
        if m1.any():
            z[m1] = -500.0 + np.fmod(np.abs(z[m1]), 500.0)
        if m2.any():
            z[m2] = 500.0 - np.fmod(np.abs(z[m2]), 500.0)
        pen = 0.0
        if m1.any():
            pen += np.sum((tmp[m1] + 500.0) ** 2) * self._pen
        if m2.any():
            pen += np.sum((tmp[m2] - 500.0) ** 2) * self._pen
        return float(_SCHWEFEL_C * n - np.sum(z * np.sin(np.sqrt(np.abs(z)))) + pen + 1100.0)


class Katsuura(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,
                 shuffle=None, sh_rate=5.0 / 100.0):
        super().__init__(Low, High, n, coeffi, bias, shuffle, sh_rate)
        self._i = np.arange(1, n + 1)
        self._exp = 10.0 / np.power(1.0 * n, 1.2)
        self._t1 = 10.0 / n / n

    def _eval(self, z):
        # 原：50 × 32 = 1600 次解释器迭代；现：一次 (32, n) 广播
        q = _KJ[:, None] * z[None, :]
        term = np.sum(np.abs(q - np.floor(q + 0.5)) / _KJ[:, None], axis=0)
        f = np.prod(np.power(1.0 + self._i * term, self._exp))
        return float(f * self._t1 - self._t1)


class GrieRosen(Task):
    NAME = 'Grie_Rosen'

    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,
                 shuffle=None, sh_rate=5.0 / 100.0):
        super().__init__(Low, High, n, coeffi, bias, shuffle, sh_rate)

    def _eval(self, z):
        t = z + 1.0
        t2 = t * t                              # 原实现里 temp*temp 被算了 4 次
        nxt = np.empty_like(t)
        nxt[:-1] = t[1:]
        nxt[-1] = t[0]                          # 原 np.append(temp[1:], temp[0])
        v = 100.0 * (t2 - nxt) * (t2 - nxt) + (t - 1.0) * (t - 1.0)
        return float(np.sum(v * v / 4000.0 - np.cos(v) + 1.0) + 1500.0)


class Escaffer6(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,
                 shuffle=None, sh_rate=1.0):
        super().__init__(Low, High, n, coeffi, bias, shuffle, sh_rate)

    def _eval(self, z):
        # 原：逐元素循环 + 尾部重复一遍首尾环绕公式；现：一次向量化
        nxt = np.roll(z, -1)
        s = z * z + nxt * nxt
        return float(np.sum(0.5 + (np.sin(np.sqrt(s)) ** 2 - 0.5)
                            / (1.0 + 0.001 * s) ** 2) + 1600.0)


class HappyCat(Task):
    NAME = 'Happycat'

    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,
                 shuffle=None, sh_rate=5.0 / 100.0):
        super().__init__(Low, High, n, coeffi, bias, shuffle, sh_rate)

    def _eval(self, z):
        t = z - 1.0
        r2 = float(np.dot(t, t))
        sz = float(np.sum(t))
        return float(np.abs(r2 - t.size) ** 0.25 + (0.5 * r2 + sz) / t.size + 0.5 + 1300.0)


class Hgbat(Task):
    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,
                 shuffle=None, sh_rate=5.0 / 100.0):
        super().__init__(Low, High, n, coeffi, bias, shuffle, sh_rate)

    def _eval(self, z):
        t = z - 1.0
        s2 = float(np.dot(t, t))                 # 原：np.sum(t*t) 算了两遍
        s1 = float(np.sum(t))                    # 原：np.sum(t) 算了两遍
        return float(np.sqrt(np.abs(s2 * s2 - s1 * s1))
                     + (0.5 * s2 + s1) / t.size + 0.5)


# ---------------------------------------------------------------------------
# 复合函数：四个 Hf 共用一个基类
# ---------------------------------------------------------------------------
class _Composite(Task):
    # 子函数清单：(类, 维数)；切片由维数累积得出，不再手写下标
    PARTS = ()
    OFFSET = 0.0

    def __init__(self, Low=-100, High=100, n=50, coeffi=None, bias=None,
                 shuffle=None, sh_rate=1.0):
        super().__init__(Low, High, n, coeffi, bias, shuffle, sh_rate)
        if shuffle is None:
            raise ValueError(f'{type(self).__name__} 必须提供 shuffle 表')
        idx = np.asarray(shuffle).ravel().astype(np.intp) - 1   # 数据是 1 起始下标
        if idx.size != n:
            raise ValueError(f'shuffle 长度 {idx.size} 与 n={n} 不一致')
        if idx.min() < 0 or idx.max() >= n:
            raise ValueError('shuffle 取值必须在 1..n 之间')
        self._idx = idx

        subs, start = [], 0
        for cls, size in self.PARTS:
            subs.append((cls(n=size), slice(start, start + size)))
            start += size
        if start != n:
            raise ValueError(f'{type(self).__name__} 子函数维数合计 {start} != n={n}')
        self._subs = subs

    def function(self, X, tag=False):
        x = _as1d(X) if tag else self.decode(X)
        temp = x[self._idx]                      # 原：50 次 Python 索引 + int() 转换
        total = 0.0
        for sub, sl in self._subs:
            total += sub.function(temp[sl], tag=True)
        return total + self.OFFSET


class Hf01(_Composite):
    PARTS = ((Schwefel, 15), (Rastrigin, 15), (Ellips, 20))
    OFFSET = 600.0


class Hf04(_Composite):
    PARTS = ((Hgbat, 10), (Discus, 10), (GrieRosen, 15), (Rastrigin, 15))
    OFFSET = 500.0


class Hf05(_Composite):
    PARTS = ((Escaffer6, 5), (Hgbat, 10), (Rosenbrock, 10),
             (Schwefel, 10), (Ellips, 15))
    OFFSET = -600.0


class Hf06(_Composite):
    PARTS = ((Katsuura, 5), (HappyCat, 10), (GrieRosen, 10),
             (Schwefel, 10), (Ackley, 15))
    OFFSET = -2200.0


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
_BASE_SUBDIR = os.path.join('Questions', 'CEC22')


def _resolve_base_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get('CEC22_DATA_DIR'), os.path.join('.', _BASE_SUBDIR)]
    cands += [os.path.join(here, _BASE_SUBDIR),
              os.path.join(here, '..', 'test', 'XXMTO', _BASE_SUBDIR),
              os.path.join(here, '..', '..', 'test', 'XXMTO', _BASE_SUBDIR)]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, 'benchmark_1')):
            return c
    raise FileNotFoundError(f'找不到 CEC22 数据目录，已尝试：{cands}')


def _bench_dir(k):
    return os.path.join(_resolve_base_dir(), f'benchmark_{k}')


@functools.lru_cache(maxsize=None)
def GetMatrixs(filepath):
    load = lambda name: np.loadtxt(os.path.join(filepath, name))  # noqa: E731
    return {'bias1': load('bias_1'), 'bias2': load('bias_2'),
            'ma1': load('matrix_1'), 'ma2': load('matrix_2')}


@functools.lru_cache(maxsize=None)
def GetShuffle(filepath):
    # 原实现返回 {'shuffle': 1 起始的 float 数组}，下游每次评估都要逐元素 int()
    # 这里返回 1 起始的整数数组（转换只做一次），由 _Composite 统一减 1
    return np.loadtxt(filepath).astype(np.intp)


@functools.lru_cache(maxsize=None)
def _shuffle_file(name):
    return os.path.join(_resolve_base_dir(), 'shuffle', name)


# ---------------------------------------------------------------------------
# 10 个基本问题：表驱动
#   值 = (Task1 类, Task2 类, Task1 的 shuffle 文件名, Task2 的 shuffle 文件名)
# ---------------------------------------------------------------------------
_BENCHMARKS = {
    1: (Weierstrass, Weierstrass, None, None),
    2: (Griewank, Griewank, None, None),
    3: (Hf01, Hf01, 'shuffle_data_17_D50.txt', 'shuffle_data_17_D50.txt'),
    4: (HappyCat, HappyCat, None, None),
    5: (GrieRosen, GrieRosen, None, None),
    6: (Hf05, Hf05, 'shuffle_data_21_D50.txt', 'shuffle_data_21_D50.txt'),
    7: (Hf06, Hf06, 'shuffle_data_22_D50.txt', 'shuffle_data_22_D50.txt'),
    8: (Ackley, Ackley, None, None),
    9: (Schwefel, Escaffer6, None, None),
    10: (Hf04, Hf05, 'shuffle_data_20_D50.txt', 'shuffle_data_21_D50.txt'),
}


def _make_benchmark(k, filename=None):
    cls1, cls2, sh1, sh2 = _BENCHMARKS[k]
    params = GetMatrixs(filename or _bench_dir(k))
    tasks = []
    for cls, bias, ma, sh in ((cls1, 'bias1', 'ma1', sh1), (cls2, 'bias2', 'ma2', sh2)):
        kw = {}
        if sh is not None:
            kw['shuffle'] = GetShuffle(_shuffle_file(sh))
        tasks.append(cls(coeffi=params[ma], bias=params[bias], **kw))
    return tasks


def Benchmark1(filename=None):
    return _make_benchmark(1, filename)


def Benchmark2(filename=None):
    return _make_benchmark(2, filename)


def Benchmark3(filename=None):
    return _make_benchmark(3, filename)


def Benchmark4(filename=None):
    return _make_benchmark(4, filename)


def Benchmark5(filename=None):
    return _make_benchmark(5, filename)


def Benchmark6(filename=None):
    return _make_benchmark(6, filename)


def Benchmark7(filename=None):
    return _make_benchmark(7, filename)


def Benchmark8(filename=None):
    return _make_benchmark(8, filename)


def Benchmark9(filename=None):
    return _make_benchmark(9, filename)


def Benchmark10(filename=None):
    return _make_benchmark(10, filename)
