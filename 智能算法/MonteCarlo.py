import numpy as np
# from pairedpermtest.utils import exact_acc, exact_f1, f1
from numba import jit

from itertools import product
def exact_acc(xs, ys, statistic=np.mean):
    """
    Compute the exact p-value for the paired permutation test for accuracy.
    This uses a brute force method.
    WARNING: this is slow and should only be used for testing purposes
    :param xs: accuracy scores for system A
    :param ys: accuracy scores for system B
    :param statistic: accumulation statistic for accuracy (default mean)
    :return: exact p-value on the paired permutation test between xs and ys
    """
    def effect(xs,ys): return np.abs(statistic(xs) - statistic(ys))
    observed = effect(xs, ys)
    p = 0.0
    n = len(xs)
    pe = 2 ** -n
    for swaps in product(*([0,1] for _ in range(n))):
        swaps = np.array(swaps, dtype=bool)
        E = effect(np.select([swaps,~swaps],[xs,ys]),  # swap elements accordingly
                   np.select([~swaps,swaps],[xs,ys]))
        p += pe * (E >= observed)
    return p

@jit(nopython=True, nogil=True, cache=True)
def f1(xs):
    """
    Compute the F1 score of a system.
    :param xs: [N, 2] array containing the number of true positives of each
    prediction in xs[:, 0] and number of incorrect predictions in xs[:, 1]
    :return: F1 score of xs
    """
    tp = np.sum(xs[:, 0])
    incorrect = np.sum(xs[:, 1])
    if tp + incorrect == 0:
        return 0.
    return tp / (tp + 0.5 * incorrect)

def exact_f1(xs, ys):
    """
    Compute the exact p-value for the paired permutation test for F!.
    This uses a brute force method.
    WARNING: this is slow and should only be used for testing purposes
    Both xs and ys are [N, 2] arrays containing the number of true positives of each
    prediction in xs[:, 0] (or ys[:, 0] and number of incorrect predictions in xs[:, 1] (or ys[:, 1]
    :param xs: accuracy scores for system A
    :param ys: accuracy scores for system B
    :return: exact p-value on the paired permutation test between xs and ys
    """
    def effect(xs, ys): return np.abs(f1(xs) - f1(ys))
    observed = effect(xs, ys)
    p = 0.0
    n = len(xs)
    pe = 2 ** -n
    for swaps in product(*([0, 1] for _ in range(n))):
        swaps = np.array(swaps, dtype=bool).repeat(2).reshape(n, 2)
        E = effect(
            np.where(swaps, xs, ys),  # swap elements accordingly
            np.where(~swaps, xs, ys),
        )
        p += pe * (E >= observed)
    return p


@jit(nopython=True, nogil=True, cache=True)
def random_swap(xs, ys):
    """
    Create a random pair of samples, (xs_, ys_), from (xs, ys).
    For element i:
        - xs_[i] = xs[i] and ys_[i] = ys[i] with probability 0.5
        - xs_[i] = ys[i] and ys_[i] = ss[i] with probability 0.5
    :param xs: scores of system A
    :param ys: scores of system B
    :param k:
    :return: paired samples (xs_, ys_) as in description
    """
    n = len(xs)             # n 为xs矩阵第一个轴的维度
    k = xs.shape[-1] if len(xs.shape) > 1 else 1
    # xs.shape[-1] 获得xs矩阵形状的最后一个轴的维度；如果xs的轴数大于1，则k为最后一个轴的维度；否则（即xs为一个向量或标量）就设为1
    swaps = (np.random.random(n) < 0.5).repeat(k).reshape(n, k)
    # ~swaps对swaps取反
    xs_ = np.select([swaps, ~swaps], [xs.reshape(n, k), ys.reshape(n, k)])
    ys_ = np.select([~swaps, swaps], [xs.reshape(n, k), ys.reshape(n, k)])
    # 随机交换xs和ys中的值，对于
    return xs_, ys_


@jit(nopython=True, nogil=True, cache=True)
def monte_carlo(xs, ys, K, effect):
    """
    Runs monte carlo sampling approximation of the paired permutation test
    based on score function effect with K samples
    :param xs: scores of system A
    :param ys: scores of system B
    :param K: number of samples
    :param effect: Scoring function between xs and ys
    :return: Approximate p-value of the paired permutation test
    """
    p_val = 0
    obs = effect(xs, ys)
    for _ in range(K):
        xs_, ys_ = random_swap(xs, ys)
        if effect(xs_, ys_) >= obs:
            p_val += 1
    return p_val / K


@jit(nopython=True, nogil=True, cache=True)
def acc_diff(xs, ys):
    """
    Returns absolute difference in accuracy between two systems
    :param xs: accuracy scores of system A
    :param ys: accuracy scores of system B
    :return: absolute difference in mean scores
    """
    return np.abs(np.mean(xs) - np.mean(ys))


def perm_test_acc(xs, ys, K):
    """
    Runs monte carlo sampling approximation of the paired permutation test
    based on difference of accuracy with K samples
    :param xs: scores of system A
    :param ys: scores of system B
    :param K: number of samples
    :return: Approximate p-value of the paired permutation test
    """
    return monte_carlo(xs, ys, K, acc_diff)


@jit(nopython=True, nogil=True, cache=True)
def f1_diff(xs, ys):
    """
    Returns absolute difference in F1 scores between two systems
    :param xs: accuracy scores of system A
    :param ys: accuracy scores of system B
    :return: absolute difference in F1 scores
    """
    return np.abs(f1(xs) - f1(ys))


def perm_test_f1(xs, ys, K):
    """
    Runs monte carlo sampling approximation of the paired permutation test
    based on difference of F1 scores with K samples
    :param xs: scores of system A
    :param ys: scores of system B
    :param K: number of samples
    :return: Approximate p-value of the paired permutation test
    """
    return monte_carlo(xs, ys, K, f1_diff)


def _test_acc():
    N = 10
    C = 5
    K = 10000
    xs = np.random.randint(0, C, N)
    ys = np.random.randint(0, C, N)
    x = perm_test_acc(xs, ys, K)
    y = exact_acc(xs, ys)
    assert np.isclose(x, y, rtol=5e-1), f"{y} =!= {x}"


def _test_f1():
    N = 10
    C = 5
    K = 10000
    xs = np.random.randint(0, C, (N, 2))
    ys = np.random.randint(0, C, (N, 2))
    x = perm_test_f1(xs, ys, K)
    y = exact_f1(xs, ys)
    assert np.isclose(x, y, rtol=5e-1), f"{y} =!= {x}"


def tests():
    print("Testing accuracy...")
    for _ in range(20):
        _test_acc()
    print("ok")
    print("Testing f1...")
    for _ in range(20):
        _test_f1()
    print("ok")

if __name__ == '__main__':
    tests()