from math import exp, sqrt

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

test_approx_sqrt()


def approx_max(x_vec, d):
    ## 1) compute mean
    ## 2) subtract x and square
    ## 3) compute mean
    ## This gives variance. Shift of 4*variance will likely be very close to the max

    mean = sum(x_vec)/len(x_vec)
    x_sub = [pow(x - mean, 2) for x in x_vec]
    variance = sum(x_sub)/len(x_sub)

    SCALE = 10000

    # return 4*sqrt(variance)
    return 4*approx_sqrt(variance/SCALE, d) * sqrt(SCALE)

def test_approx_max():

    max_val = 100

    vals = range(1, max_val)
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
        if (b != pow(1-x, pow(2, i+1))):
            print("\t b diff:", abs(b-pow(1-x, pow(2, i+1))))
        a *= 1 + b 
        print(a)
    return a

def test_approx_inverse():

    max_val = 20

    vals = range(1, max_val)

    for v in vals:
        v = 1+ v / max_val

        correct = 1/v 

        res = approx_inv(v, d=6)

        print(v, correct, res)

# test_approx_inverse()