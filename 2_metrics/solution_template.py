from math import log2

from torch import Tensor, sort


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    _, indices = sort(ys_pred, 0, descending=True)
    ys_true_sorted = ys_true[indices]
    res = 0
    for iter in range(len(ys_true_sorted)):
        for i, val in enumerate(ys_true_sorted[:-1]):
            if ys_true_sorted[i+1] > ys_true_sorted[i]:
                res+=1
                ys_true_sorted[i+1], ys_true_sorted[i] = ys_true_sorted[i].item(), ys_true_sorted[i+1].item()
    return res


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == "const":
        return y_value
    if gain_scheme == "exp2":
        return (2 ** y_value - 1)
    pass


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    _, indices = sort(ys_pred, 0, descending=True)
    ys_true_sorted = ys_true[indices]
    res = 0
    for i, val in enumerate(ys_true_sorted):
        val = float(compute_gain(val, gain_scheme)) / log2(i+2)
        res += val
    return float(res)


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    ys_true_sorted_ideal, _ = sort(ys_true, 0, descending=True)
    ideal_dcg = 0
    for i, val in enumerate(ys_true_sorted_ideal):
        val = float(compute_gain(val, gain_scheme)) / log2(i+2)
        ideal_dcg += val
    res = dcg(ys_true, ys_pred, gain_scheme)
    return (res / float(ideal_dcg))


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    if sum(ys_true) < 1:
        return -1.0
    _, indices = sort(ys_pred, 0, descending=True)
    ys_true_sorted = ys_true[indices][:k]
    precision = float(sum(ys_true_sorted)) / min(k, sum(ys_true))
    return float(precision)


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    _, indices = sort(ys_pred, 0, descending=True)
    return 1.0 / ((ys_true[indices] == 1).nonzero()[0].item() + 1)


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    _, indices = sort(ys_pred, 0, descending=True)
    ys_true_sorted = ys_true[indices]
    p_look = 1
    p_rel = ys_true_sorted[0].item()
    p_found = p_look * p_rel
    for i in range(len(ys_true_sorted) - 1):
        p_look = p_look * (1 - p_rel) * (1 - p_break)
        p_rel = ys_true_sorted[i+1].item()
        p_found += p_look * p_rel
    return p_found


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    if sum(ys_true) < 1:
        return -1.0
    _, indices = sort(ys_pred, 0, descending=True)
    ys_true_sorted = ys_true[indices]
    sum_ = 0 
    rolling_sum = 0
    for i in range(len(ys_true_sorted)):
        if ys_true_sorted[i] == 1:
            sum_ += ys_true_sorted[i].item()
            rolling_sum += sum_ / (i+1)
    return float(rolling_sum) / float(sum(ys_true))