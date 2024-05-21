from math import exp, sqrt, ceil, log2
import scipy.special

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


def approx_compare(x, d=2):
    ## always just compare with zero. Plaintext offset can be added for zero error

    res = x 
    d_g = d
    d_f = d

    total_depth = 0

    for _ in range(d_g):
        res, depth = compare_g(res, 4)
        total_depth += depth
    for _ in range(d_f):
        res, depth = compare_f(res, 4)
        total_depth += depth

    return res  # , total_depth


def test_approx_compare():
    vals = range(-5, 5)
    for i in vals:
        for j in vals:
            # print(i-j, approx_compare((i-j)/10), (i-j))
            assert(abs(max(i,j) - (j + (approx_compare((i-j)/10)+1)/2*(i-j))) < 1e-10)

test_approx_compare()

def approx_sqrt(x, d):
    assert(0 <= x)
    assert(x <= 1)
    a = x
    b = x-1
    for _ in range(d):
        a *= 1 - b/2
        b = pow(b, 2) * (b - 3)/4
    return a

def test_approx_sqrt():

    max_val = 20

    vals = range(1, max_val)

    for v in vals:
        v /= max_val

        correct = sqrt(v)

        res = approx_sqrt(v, d=6)

        print(v, correct, res)

# test_approx_sqrt()


def approx_max(x_vec, d):
    ## 1) compute mean
    ## 2) subtract x and square
    ## 3) compute mean
    ## This gives variance. Shift of 4*variance will likely be very close to the max

    SCALE = 10
    
    i = 0
    j = 0
    if len(x_vec) == 0:
        return
    elif len(x_vec) == 1:
        return x_vec[0]
    elif len(x_vec) == 2:
        i = x_vec[0]
        j = x_vec[1]
    else:
        midpoint = len(x_vec)//2
        i = approx_max(x_vec[:midpoint], d)
        j = approx_max(x_vec[midpoint:], d)

    return j + (i-j) * (approx_compare((i-j)/SCALE, d)+1)/2


    mean = sum(x_vec)/len(x_vec)
    x_sub = [pow(x - mean, 2) for x in x_vec]
    variance = sum(x_sub)/len(x_sub)

    SCALE = 10000

    # return 4*sqrt(variance)
    return 4*approx_sqrt(variance/SCALE, d) * sqrt(SCALE)

def test_approx_max():

    # max_val = 100

    # vals = range(1, max_val)
    vals = [0.0307, -1.3192,  0.0000,  0.0000,  0.0000]
    # vals = [v* for v in vals]

    correct = max(vals)

    res = approx_max(vals, 4)

    print(correct, res)

test_approx_max()



def approx_exp(x, r = 6):
    # PARAM_R = 6
    output = pow(1 + x/pow(2, r), pow(2, r))
    # Set output of -inf entries to 0
    # -inf is sometimes (always?) equal to -3.4028e+38
    # output[x <= -3.4028e+38] = 0
    return output

def test_approx_exp():

    max_val = 20

    vals = range(1, max_val)

    for v in vals:
        v -= max_val

        correct = exp(v)

        res = approx_exp(v, r=6)

        print(v, correct, res)

# test_approx_exp()


def approx_inv(x, d=5):
    assert(0 < x)
    assert(x < 2)
    a = 2 - x
    b = 1 - x 
    for i in range(d):
        b = pow(b, 2)
        # assert(b == pow(1-x, pow(2, i+1)))
        # if (b != pow(1-x, pow(2, i+1))):
        #     print("\t b diff:", abs(b-pow(1-x, pow(2, i+1))))
        a *= 1 + b 
        # print(a)
    return a

# def goldschmidt(x, d=5):
#     for i in range(d):
#         """
#         TODO: Implement this
#         """


def test_approx_inverse():

    max_val = 20

    vals = range(1, max_val)

    for v in vals:
        v = 1+ v / max_val

        correct = 1/v 

        res = approx_inv(v, d=6)
        # res2 = goldschmidt(v, d=6)

        print(v, correct)
        # print("\t", res, res2)

# test_approx_inverse()


def correct_softmax(x_vec):
    ## compute the max
    x_max = max(x_vec)
    print("correct max", x_max)
    ## exponentiate
    x_exp = [exp(x - x_max) for x in x_vec]
    print("correct exp", x_exp)
    ## sum
    x_sum = sum(x_exp)
    print("correct sum", x_sum)
    ## invert
    print("correct inv", 1.0/x_sum)
    return [x/x_sum for x in x_exp]

def approx_softmax(x_vec, d_max, d_exp, d_inv):
    x_max = approx_max(x_vec, d_max)
    print("res max", x_max)
    x_exp = [approx_exp(x - x_max, d_exp) for x in x_vec]
    print("res exp", x_exp)
    x_sum = sum(x_exp)
    print("res sum", x_sum)
    # print(x_sum)
    SCALE = 5
    x_inv = approx_inv(x_sum/SCALE, d_inv) / SCALE
    print("res inv", x_inv)
    return [x * x_inv for x in x_exp]

def test_softmax():
    vals = [0.0307, -1.3192,  0.0000,  0.0000,  0.0000]
    
    correct = correct_softmax(vals)

    d_max = 6
    d_exp = 6
    d_inv = 6

    res = approx_softmax(vals, d_max, d_exp, d_inv)

    print(correct)
    print(res)

# test_softmax()
