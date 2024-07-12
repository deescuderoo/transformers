from math import exp, sqrt, ceil, log2
import scipy.special
import numpy as np


class Poly:
    def __init__(self, coeffs):
        self.coeffs = coeffs.copy()
        self.degree = len(self.coeffs)-1

    def copy(self):
        return Poly(self.coeffs)

    def zeros(degree):
        coeffs = [0]*(degree+1)
        return Poly(coeffs)
    
    def ones(degree):
        coeffs = [1]*(degree+1)
        return Poly(coeffs)

    def oneHot(i, degree):
        coeffs = [0]*(degree+1)
        coeffs[i] = 1
        return Poly(coeffs)

    # def random():
    #     coeffs = [0]*n
    #     for i in range(n):
    #         coeffs[i] = randint(0, p-1)
    #     return Poly(coeffs)

    def toString(self):
        res = ""
        for c in self.coeffs:
            res += str(c)
            res += ", "
        return "[" + res[:-2] + "]"

    def eval(self, x):
        res = 0
        x_pow = 1
        for c in self.coeffs:
            res += (x_pow * c)
            x_pow = (x_pow * x)
        return res
    
    def compose(self, input):
        res = Poly([0])
        in_pow = Poly([1])
        for c in self.coeffs:
            res = res + in_pow.scalMult(c)
            in_pow = (in_pow * input)
        return res

    def copy_and_pad(input, length):
        res = []
        for c in input:
            res.append(c)
        while len(res) < length:
            res.append(0)
        return res

    def add(p1, p2):
        max_length = max(len(p1.coeffs), len(p2.coeffs))
        p1_coeffs = Poly.copy_and_pad(p1.coeffs, max_length)
        p2_coeffs = Poly.copy_and_pad(p2.coeffs, max_length)
        res_coeffs = [(x+y) for x,y in zip(p1_coeffs, p2_coeffs)]
        return Poly(res_coeffs)

    def __add__(self, p2):
        return Poly.add(self, p2)

    def sub(p1, p2):
        max_length = max(len(p1.coeffs), len(p2.coeffs))
        p1_coeffs = Poly.copy_and_pad(p1.coeffs, max_length)
        p2_coeffs = Poly.copy_and_pad(p2.coeffs, max_length)
        res_coeffs = [(x-y) for x,y in zip(p1_coeffs, p2_coeffs)]
        return Poly(res_coeffs)

    def __sub__(self, p2):
        return Poly.sub(self, p2)

    def mult(p1, p2):
        total_degree = p1.degree + p2.degree
        res_coeffs = [0]*(total_degree+1)
        for i in range(len(p1.coeffs)):
            for j in range(len(p2.coeffs)):
                res_coeffs[i+j] += p1.coeffs[i]*p2.coeffs[j]

        return Poly(res_coeffs)

    def __eq__(self, p2):
        if (len(self.coeffs) != len(p2.coeffs)):
            return False
        for x,y in zip(self.coeffs, p2.coeffs):
            if x != y:
                return False 
        return True

    def __mul__(self, p2):
        return Poly.mult(self, p2)

    def scalMult(self, v):
        res_coeffs = [(v*c) for c in self.coeffs]
        return Poly(res_coeffs)


def compare_f(x, n=4):
    res = 0
    for i in range(n+1):
        ## c * x * (1 - x^2)^i
        res += 1/pow(4, i) * scipy.special.comb(2*i, i) * x * pow(1 - pow(x, 2), i)

    return res, ceil(log2(n))+1

def get_poly_f(n=4):
    res = Poly.zeros(0)
    x_poly = Poly([0,1])
    one_minus_x_2_poly = Poly([1,0,-1])
    one_minus_x_2_poly_pow = Poly([1])
    for i in range(n+1):
        ## c * x * (1 - x^2)^i
        # res += 1/pow(4, i) * scipy.special.comb(2*i, i) * x * pow(1 - pow(x, 2), i)
        coeff = Poly([1/pow(4, i) * scipy.special.comb(2*i, i)])
        toAdd = coeff * x_poly 
        toAdd = toAdd * one_minus_x_2_poly_pow
        res = res + toAdd 

        one_minus_x_2_poly_pow = one_minus_x_2_poly_pow * one_minus_x_2_poly

    return res

def test_expanded_f():
    poly_f = get_poly_f()
    print("f", poly_f.coeffs)

    max = 10
    for i in range(-max, max):
        x = float(i) / float(max)

        correct = compare_f(x)
        res = poly_f.eval(x)

        # print(x, correct[0], res)
        assert(abs(correct[0]-res) < 1e-10)

test_expanded_f()


def compare_g(x, index = 4):
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

def get_poly_g(index = 4):
    return Poly([0, 5850/pow(2, 10), 0, -34974/pow(2, 10), 0, 97015/pow(2, 10), 0, -113492/pow(2, 10), 0, 46623/pow(2, 10)]) 

def test_expanded_g():
    poly_g = get_poly_g(4)
    print("g", poly_g.coeffs)

    max = 10
    for i in range(-max, max):
        x = float(i) / float(max)

        correct = compare_g(x)
        res = poly_g.eval(x)

        # print(x, correct[0], res)
        assert(abs(correct[0]-res) < 1e-10)

test_expanded_g()

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
    # print("approx compare depth", total_depth, d)
    return res  # , total_depth

def get_full_comparison_poly(d=2):
    poly_f = get_poly_f()
    poly_g = get_poly_g()

    res = poly_g 
    for _ in range(1, d):
        res = poly_g.compose(res)

    for _ in range(d):
        res = poly_f.compose(res)

    return res


def get_cheby_polys():
    cheb = np.polynomial.chebyshev.Chebyshev((0,0,0,0,0,1))
    coef = np.polynomial.chebyshev.cheb2poly(cheb.coef)

    print(coef)

    back_to_cheb = np.polynomial.chebyshev.poly2cheb(coef)

    print(back_to_cheb)

# get_cheby_polys()


def get_comparison_cheb():
    poly_f = get_poly_f()
    poly_g = get_poly_g()

    print(poly_f.coeffs)
    print(poly_g.coeffs)

    f = np.polynomial.polynomial.Polynomial(tuple(poly_f.coeffs))
    g = np.polynomial.polynomial.Polynomial(tuple(poly_g.coeffs))

    print(f)
    print(g)

    # f_circ_g = np.polyval(f, g)
    f_circ_g = f(f(g(g)))

    print(f_circ_g)

    composed_cheb = np.polynomial.chebyshev.poly2cheb(f_circ_g)

    print(composed_cheb)

# get_comparison_cheb()

def test_compose():
    poly_f = get_poly_f()
    poly_g = get_poly_g()

    poly_composed = poly_f.compose(poly_g)

    print(poly_f.coeffs)
    print(poly_g.coeffs)
    print(poly_composed.coeffs)

    max = 10
    for i in range(-max, max):
        x = float(i) / float(max)

        # correct = approx_compare(x, 1)
        correct = poly_f.eval(poly_g.eval(x))
        res = poly_composed.eval(x)

        print(x, correct, res)
        assert(abs(correct-res) < 1e-10)

# test_compose()

def test_expanded_compare():
    poly_compare = get_full_comparison_poly(1)
    print(poly_compare.coeffs)

    max = 10
    for i in range(-max, max):
        x = float(i) / float(max)

        correct = approx_compare(x, 1)
        res = poly_compare.eval(x)

        print(x, correct, res)
        assert(abs(correct-res) < 1e-10)

# test_expanded_compare()