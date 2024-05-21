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
    x[x <= -3.4028e+37] = 0
    maxes = torch.max(x, dim, keepdim=True)[0]
    x_exp = torch.exp(x-maxes)
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    return x_exp/x_exp_sum


def approx_softmax(x, dim=None):
    x[x <= -3.4028e+37] = 0
    # print(f"max:\n{x.max()}")
    # maxes = torch.max(x, dim, keepdim=True)[0]
    # maxes = torch.max(x, dim, keepdim=True)
    # print("correct max", maxes)
    print("input", x[0, 0, 1])
    maxes = approx_max(x, 6)
    # print("res maxes", maxes[0, 0, 1])
    # correct_max = torch.max(x, dim, keepdim=True)[0]
    # print(maxes, correct_max)
    # assert(maxes == correct_max)
    # maxes = torch.empty(x.dim()).fill_(approx_max(x, 6))
    maxes = maxes.unsqueeze(-1)

    maxes_extended = maxes 
    # for _ in range(1, x_vec.shape[-1]):
    for _ in range(1, x.shape[-1]):
        maxes_extended = torch.cat((maxes_extended, maxes), -1)
    
    # print(x.shape)
    # print(maxes_extended.shape)

    # # For debugging
    # if not torch.all(x-maxes <= 0):
    #     print(f"error:\n{(approx_exp(x-maxes) - torch.exp(x-maxes)).abs().max()}")
    #     print(f"input:\n{x-maxes}")
    #     print(f"real:\n{torch.exp(x-maxes)}")
    #     print(f"approx:\n{approx_exp(x-maxes)}")
    #     print("-----------------------------------------\n")
    #     raise RuntimeError

    # assert(torch.all(x-maxes_extended <= 1))

    LOG_SCALE_EXP = 3
    inner_diff = (x-maxes_extended) / pow(2, LOG_SCALE_EXP)
    # if not (torch.all(inner_diff <= 1.5)):
    print("exponents: ", inner_diff[0, 1, 1])
    #     # print(torch.index(inner_diff <= 1))
    #     # bad_index = torch.all(inner_diff <= 1)
    #     # print(inner_diff[.nonzero()])
        # assert(False)

    x_exp = approx_exp(inner_diff, 6)   ## all exponents are < 0
    # for _ in range(LOG_SCALE_EXP):
        # x_exp *= x_exp
    # x_exp = approx_exp(-1*x)   ## all exponents are < 0
    # x_exp = torch.exp(x-maxes)
    print("res exp", x_exp[0, 1, 1])


    # x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    x_exp_sum = torch.sum(x_exp , dim=-1)  # [0]/x_vec.dim()
    
    print("res sum", x_exp_sum[0, 1, 1])
    # print(x_exp_sum)
    
    LOG_SCALE_INV = 3
    print(x_exp_sum)
    x_inv = approx_inv(x_exp_sum / pow(2,LOG_SCALE_INV), d=6)
    for _ in range(LOG_SCALE_INV):
        x_inv *= x_inv

    x_inv = x_inv.unsqueeze(-1)

    x_inv_extended = x_inv
    for _ in range(1, x_inv.shape[-1]):
        x_inv_extended = torch.cat((x_inv_extended, x_inv), -1)

    print("res inv", x_inv)

    # print(x_exp_sum, 1/x_exp_sum, approx_inv(x_exp_sum / SCALE, d=10) / SCALE)
    # return x_exp/x_exp_sum
    return x_exp * x_inv_extended
