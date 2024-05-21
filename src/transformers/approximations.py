from math import ceil, tanh, log2, sqrt, pi
import scipy.special
import torch

def compare_f(x, n):
    res = 0
    for i in range(n+1):
        ## c * x * (1 - x^2)^i
        res += 1/pow(4, i) * scipy.special.comb(2*i, i) * x * pow(1 - pow(x, 2), i)

    return res, ceil(log2(n))+1


def compare_g(x, index = 1):
    if index == 1:
        return -1*1359/pow(2, 10) * pow(x, 3) + (2126)/pow(2, 10) * x, 2
    
    elif index == 2:
        return 3796/pow(2, 10) * pow(x, 5) - 6108 * pow(x, 3) + 3334/pow(2, 10) * x, 3

    elif index == 3:
        return -12860/pow(2, 10) * pow(x, 7) + 25614/pow(2, 10) * pow(x, 5) - 16577/pow(2, 10) * pow(x, 3) + 4589/pow(2, 10) * x, 4

    elif index == 4:
        return 46623/pow(2, 10) * pow(x, 9) - 113492/pow(2, 10) * pow(x, 7) + 97015/pow(2, 10) * pow(x, 5) - 34974/pow(2, 10) * pow(x, 3) + 5850/pow(2, 10) * x, 5

    else:
        print("unknown index")
        assert(False)


def approx_compare(x):
    ## always just compare with zero. Plaintext offset can be added for zero error

    res = x 
    d_g = 3
    d_f = 4

    total_depth = 0

    for _ in range(d_g):
        res, depth = compare_g(res, 4)
        total_depth += depth
    for _ in range(d_f):
        res, depth = compare_f(res, 4)
        total_depth += depth

    return res, total_depth


def test_approx():
    for x in range(-100, 100):
        x /= 100000
        print(x, approx_compare(x))


# test_approx()

def approx_less_than(x, t):
    # assert(abs(t - x) < 1)
    res, d = approx_compare(t-x) 
    # print(d)
    return (res + 1)/2


# SOFTMAX

def approx_sqrt(x, d):
    if not (torch.all(0 <= x)):
        print(x)
        print("approx sqrt 0 <= x failed")
        assert(False)
    assert(torch.all(x <= 1))
    a = x
    b = x-1
    for _ in range(d):
        a *= 1 - b/2
        b = pow(b, 2) * (b - 3)/4
    return a


def approx_max(x_vec, d):
    ## 1) compute mean
    ## 2) subtract x and square
    ## 3) compute mean
    ## This gives variance. Shift of 4*variance will likely be very close to the max

    return torch.sum(x_vec, dim=-1)/x_vec.shape[-1]

    # print("beginning max with input ", x_vec)
    # print("beginning max")

    mean = torch.sum(x_vec, dim=-1)/x_vec.shape[-1]
    # print("max mean", mean[0,0,1])
    # print(mean.shape)
    mean = mean.unsqueeze(-1)
    # print(mean)
    # print(mean.shape)
    mean_extended = mean 
    # for _ in range(1, x_vec.shape[-1]):
    for _ in range(1, x_vec.shape[-1]):
        mean_extended = torch.cat((mean_extended, mean), -1)
    
    # print(mean_extended)
    # print(mean_extended.shape)
    # print(x_vec.shape)
    # mean.transpose()
    # print(mean)
    # mean = mean.unsqueeze(1).repeat(1, 1, 4)
    # print(mean)
    # x_sub = [pow(x - mean, 2) for x in x_vec]
    x_sub = torch.pow(x_vec - mean_extended, 2)
    # print("max sub", x_sub[0,0,1])
    variance = torch.sum(x_sub, dim=-1)/x_vec.shape[-1]

    SCALE = 10000

    # return 4*sqrt(variance)
    return 5*approx_sqrt(variance/SCALE, d) * sqrt(SCALE)

def approx_exp(x, r = 6):
    # return torch.exp(x)
    # PARAM_R = 6
    output = pow(1 + x/pow(2, r), pow(2, r))
    # Set output of -inf entries to 0
    # -inf is sometimes (always?) equal to -3.4028e+38
    output[x <= -3.4028e+37] = 0
    return output

def approx_inv(x, d=5):
    # assert(0 < x)
    # assert(torch.all(x > 0))
    if (not torch.all(x > 0)):
        print(x)
        print("approx inv x > 0 check failed")
        assert(False)
    # assert(torch.all(x < 2))
    if (not torch.all(x < 2)):
        print(x)
        print("approx inv x < 2 check failed")
        assert(False)
    a = 2 - x
    b = 1 - x 
    for _ in range(d):
        b = pow(b, 2)
        a *= 1 + b 
    return a

def ref_softmax(x, dim=None):
    # x[x <= -3.4028e+37] = 0
    maxes = torch.max(x, dim, keepdim=True)[0]
    x_exp = torch.exp(x-maxes)
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    return x_exp/x_exp_sum


def approx_softmax(x, dim=None):

    maxes = torch.max(x, dim, keepdim=True)[0]

    # For debugging
    if not torch.all(x-maxes <= 0):
        print(f"error:\n{(approx_exp(x-maxes) - torch.exp(x-maxes)).abs().max()}")
        print(f"input:\n{x-maxes}")
        print(f"real:\n{torch.exp(x-maxes)}")
        print(f"approx:\n{approx_exp(x-maxes)}")
        print("-----------------------------------------\n")
        raise RuntimeError

    x_exp = approx_exp(x-maxes)
    # x_exp = torch.exp(x-maxes)

    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    # return x_exp/x_exp_sum

    ## TODO: Implement scaling down by the length of the sum
    x_inv = approx_inv(x_exp_sum/1000, d=6)/1000

    return x_exp * x_inv