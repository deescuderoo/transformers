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


def approx_compare(x, d_g = 4, d_f = 4):
    ## always just compare with zero. Plaintext offset can be added for zero error

    res = x 
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

def approx_less_than(x, t, d_g = 3, d_f = 4, SCALE = 1):
    # assert(abs(t - x) < 1)
    res, d = approx_compare((t-x)/SCALE, d_g=d_g, d_f=d_f) 
    # print(d)
    return (res + 1)/2


# SOFTMAX

# def approx_sqrt(x, d):
#     if not (torch.all(0 <= x)):
#         print(x)
#         print("approx sqrt 0 <= x failed")
#         assert(False)
#     assert(torch.all(x <= 1))
#     a = x
#     b = x-1
#     for _ in range(d):
#         a *= 1 - b/2
#         b = pow(b, 2) * (b - 3)/4
#     return a


def approx_max(x_vec, d_g = 3, d_f = 4):
    ## 1) compute mean
    ## 2) subtract x and square
    ## 3) compute mean
    ## This gives variance. Shift of 4*variance will likely be very close to the max

    SCALE = 1000

    i = 0
    j = 0
    if len(x_vec) == 0:
        return None
        # raise Exception("should never have emp")
    elif len(x_vec) == 1:
        return x_vec[0]
    elif len(x_vec) == 2:
        i = x_vec[0]
        j = x_vec[1]
    else:
        midpoint = len(x_vec)//2
        i = approx_max(x_vec[:midpoint], d_g, d_f)
        j = approx_max(x_vec[midpoint:], d_g, d_f)

        if i == None:
            return j 
        if j == None:
            return i
    
    # print("computing comparison in recursive max")
    # print(i.shape, j.shape)
    # print(i[1], j[1])
    i_less_than_j = approx_less_than(i, j, d_g, d_f, SCALE)
    # print(i_less_than_j[1])
    # return j + (i-j) * (approx_compare((i-j)/SCALE, d_g, d_f)+1)/2
    res = i + (j-i) * i_less_than_j
    # print(res[1])
    return res

    # # return torch.sum(x_vec, dim=-1)/x_vec.shape[-1]

    # # print("beginning max with input ", x_vec)
    # # print("beginning max")

    # mean = torch.sum(x_vec, dim=-1)/x_vec.shape[-1]
    # # print("max mean", mean[0,0,1])
    # # print(mean.shape)
    # mean = mean.unsqueeze(-1)
    # # print(mean)
    # # print(mean.shape)
    # mean_extended = mean 
    # # for _ in range(1, x_vec.shape[-1]):
    # for _ in range(1, x_vec.shape[-1]):
    #     mean_extended = torch.cat((mean_extended, mean), -1)
    
    # # print(mean_extended)
    # # print(mean_extended.shape)
    # # print(x_vec.shape)
    # # mean.transpose()
    # # print(mean)
    # # mean = mean.unsqueeze(1).repeat(1, 1, 4)
    # # print(mean)
    # # x_sub = [pow(x - mean, 2) for x in x_vec]
    # x_sub = torch.pow(x_vec - mean_extended, 2)
    # # print("max sub", x_sub[0,0,1])
    # variance = torch.sum(x_sub, dim=-1)/x_vec.shape[-1]

    # SCALE = 10000

    # # return 4*sqrt(variance)
    # return 5*approx_sqrt(variance/SCALE, d) * sqrt(SCALE)

def approx_exp(x, r = 6):
    # return torch.exp(x)
    output = pow(1 + x/pow(2, r), pow(2, r))
    # Set output of -inf entries to 0
    # -inf is sometimes (always?) equal to -3.4028e+38
    # output[x <= -3.4028e+37] = 0
    # print((torch.exp(x) - output).abs().mean())
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

def approx_div(x, y, n):
    INITIAL_ESTIMATE = 10
    F = INITIAL_ESTIMATE
    N = x
    D = y
    for _ in range(n):
        F = 2-D 
        N *= F 
        D *= F

    return N

def ref_softmax(x, dim=None):
    # x[x <= -3.4028e+37] = 0
    print("input shape", x.shape)
    maxes = torch.max(x, dim, keepdim=True)[0]
    x_exp = torch.exp(x-maxes)
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    return x_exp/x_exp_sum

def approx_softmax(x, dim=None):
    correct_maxes = torch.max(x, dim, keepdim=True)[0]
    # assert(correct_maxes == maxes)
    maxes = correct_maxes

    EXP_ITERATIONS = 10
    x_exp = approx_exp(x-maxes, EXP_ITERATIONS)
    # x_exp = torch.exp(x-maxes)
    x_exp[x <= -3.4028e+37] = 0
    # assert torch.all(x_exp <= 1)
    # x_exp = torch.exp(x-maxes)

    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)

    # return x_exp/x_exp_sum

    # Division
    # out = x_exp/x_exp_sum
    normalizer = torch.ones(x_exp.shape).sum(dim, keepdim=True)

    # assert torch.all(x_exp_sum / normalizer <= 1)

    # norm: divide by length so that quotient is <1 (denominator
    # becomes the mean)
    G_ITERATIONS = 10
    if torch.cuda.is_available():
        normalizer = normalizer.to('cuda')
        # print(f"Device: {normalizer.device}")

    out = approx_div(x_exp / normalizer, x_exp_sum / normalizer,
                     G_ITERATIONS)

    # Useful for handpicking initial approx.
    # print((1/torch.mean(x_exp, dim, keepdim=True)).mean())
    return out
