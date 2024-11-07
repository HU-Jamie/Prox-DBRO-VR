import math


def get_decaying_step(alpha, k):
    """
    Compute the decreasing learning step
    Here, we should modify the decreasing learning step into Barzilai-Borwein step-size
    :param alpha: coefficient
    :param k: iteration time
    """
    return alpha / math.sqrt(k+1)  # math.sqrt平方根


def get_decaying_step_v1(alpha, k):
    return alpha / (k + 20)




# def get_constant_step():
#    alpha = 0.001
#    return alpha