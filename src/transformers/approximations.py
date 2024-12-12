from math import ceil, tanh, log2, sqrt, pi
import scipy.special
import torch, os

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
    DEGREE_FG = 4
    res = x 
    total_depth = 0

    for _ in range(d_g):
        res, depth = compare_g(res, DEGREE_FG)
        total_depth += depth
    for _ in range(d_f):
        res, depth = compare_f(res, DEGREE_FG)
        total_depth += depth

    return res, total_depth


def test_approx():
    for x in range(-100, 100):
        x /= 100000
        print(x, approx_compare(x))


# test_approx()

def approx_less_than(x, t, d_g = 2, d_f = 2, SCALE = 1):
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

# Ensure the folder exists
OUTPUT_FOLDER = "output_cycles"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# Initialize the global variable
current_cycle = 0  # Default value is now 0

def find_max_cycle():
    global current_cycle
    max_cycle = 0
    for filename in os.listdir(OUTPUT_FOLDER):
        if filename.startswith("output_max_cycle_") and filename.endswith(".txt"):
            try:
                cycle_number = int(filename.split("_")[-1].split(".")[0])
                max_cycle = max(max_cycle, cycle_number)
            except ValueError:
                continue
    current_cycle = max_cycle

layer0_array=[1.8493261337280273, 10.321573257446289, 1.0337244272232056, 11.179471969604492, 6.565133094787598, 16.084802627563477, 3.021239757537842, 9.312780380249023, 10.720375061035156, 12.07174301147461, 10.763772964477539, 10.797805786132812]
layer1_array=[4.33711051940918, 3.385294198989868, 1.053083896636963, 2.346316337585449, 0.9965879321098328, 4.807539463043213, 4.680234909057617, 1.9316093921661377, 2.231010913848877, 2.1602818965911865, 2.092066764831543, 8.487981796264648]
layer2_array=[1.7087633609771729, 2.9535627365112305, 4.604798793792725, 2.7798538208007812, 3.2145142555236816, 3.3662657737731934, 2.259106159210205, 2.492807388305664, 3.4144763946533203, 2.8978257179260254, 2.2755203247070312, 2.1004021167755127]
layer3_array=[7.631556510925293, 7.5165534019470215, 6.794878005981445, 6.580643177032471, 6.48328971862793, 6.706857681274414, 6.219317436218262, 6.2063679695129395, 6.263098239898682, 5.790459632873535, 6.246354579925537, 6.452132225036621]
layer4_array=[3.4940876960754395, 1.149423599243164, 1.2998638153076172, 2.6921796798706055, 2.8489766120910645, 0.77414870262146, 0.46880295872688293, 4.821672439575195, 2.983339309692383, 1.9948464632034302, 2.348292112350464, 39.45146179199219]
layer5_array=[5.923620223999023, 11.5446195602417, 2.7273964881896973, 2.7468600273132324, 2.67863130569458, 7.868797302246094, 4.552934169769287, 4.86293888092041, 5.042206764221191, 5.317700386047363, 4.951260089874268, 5.159172058105469]
layer6_array=[2.7284209728240967, 2.200655460357666, 1.024259328842163, 1.5481042861938477, 3.5086355209350586, 0.7202161550521851, 2.993629217147827, 0.934133768081665, 2.356409788131714, 9.204235076904297, 4.28968620300293, 1.8297182321548462]
layer7_array=[1.9839451313018799, 6.594324588775635, 13.034424781799316, 1.9016940593719482, 2.6689841747283936, 2.9748497009277344, 2.595259666442871, 8.867525100708008, 4.641799449920654, 4.675210475921631, 11.67599868774414, 10.335289001464844]
layer8_array=[2.3718461990356445, 10.817716598510742, 4.544055938720703, 4.658476829528809, 2.6210646629333496, 1.6930010318756104, 6.571752071380615, 4.000815391540527, 4.905542373657227, 5.262595176696777, 5.410916805267334, 5.95646858215332]
layer9_array=[5.831937313079834, 6.704681396484375, 4.681728363037109, 3.184159278869629, 4.5470428466796875, 3.4491376876831055, 6.640199661254883, 3.4697933197021484, 4.080732822418213, 7.8267340660095215, 4.844199180603027, 5.132279396057129]
layer10_array=[4.52237606048584, 7.927225112915039, 6.176588535308838, 6.088727951049805, 5.833307266235352, 4.04454231262207, 5.550886631011963, 6.582158088684082, 5.128731727600098, 5.044299125671387, 5.436607837677002, 7.902202606201172]
layer11_array=[4.992722511291504, 5.467639923095703, 5.714563846588135, 4.192873001098633, 4.425979137420654, 6.185069561004639, 5.388833045959473, 6.145012855529785, 19.679534912109375, 7.636826515197754, 7.3761444091796875, 43.82441329956055]

device = torch.device("cuda")

tensors = {}
tensors[0] = torch.tensor(layer0_array, device=device).reshape(1, 12, 1, 1)
tensors[1] = torch.tensor(layer1_array, device=device).reshape(1, 12, 1, 1)
tensors[2] = torch.tensor(layer2_array, device=device).reshape(1, 12, 1, 1)
tensors[3] = torch.tensor(layer3_array, device=device).reshape(1, 12, 1, 1)
tensors[4] = torch.tensor(layer4_array, device=device).reshape(1, 12, 1, 1)
tensors[5] = torch.tensor(layer5_array, device=device).reshape(1, 12, 1, 1)
tensors[6] = torch.tensor(layer6_array, device=device).reshape(1, 12, 1, 1)
tensors[7] = torch.tensor(layer7_array, device=device).reshape(1, 12, 1, 1)
tensors[8] = torch.tensor(layer8_array, device=device).reshape(1, 12, 1, 1)
tensors[9] = torch.tensor(layer9_array, device=device).reshape(1, 12, 1, 1)
tensors[10] = torch.tensor(layer10_array, device=device).reshape(1, 12, 1, 1)
tensors[11] = torch.tensor(layer11_array, device=device).reshape(1, 12, 1, 1)

def ref_softmax(x, dim=None):
    # x[x <= -3.4028e+37] = 0
    # print("input shape", x.shape)
    maxes = torch.max(x, dim, keepdim=True)[0]
    x_exp = torch.exp(x-maxes)
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    return x_exp/x_exp_sum

def interm_softmax(x, layer_id, dim=None):
    # x[x <= -3.4028e+37] = 0
    # print("input shape", x.shape)
    # maxes = torch.max(x, dim, keepdim=True)[0]
    maxes = tensors[layer_id]
    x_exp = torch.exp(x-maxes)
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    return x_exp/x_exp_sum

def approx_softmax_store_in_file(x, layer_id, dim=None):
    #this function is used to store the max value in a file, which was later used to determine statistics
    global current_cycle
    find_max_cycle()
    print("Current cycle", current_cycle)
    output_max = os.path.join(OUTPUT_FOLDER, f"output_max_cycle_{current_cycle}.txt")
    if layer_id == 0:
        if current_cycle > 0 or os.path.exists(output_max):
            current_cycle += 1
            output_max = os.path.join(OUTPUT_FOLDER, f"output_max_cycle_{current_cycle}.txt")
        with open(output_max, 'w') as file:
            file.write(f"Cycle {current_cycle} - Layer Outputs:\n")
    print("Layer id: ", layer_id)
    print("Input shape", x.shape)
    maxes = torch.max(x, dim, keepdim=True)[0]
    print("Max shape", maxes.shape)
    with open(output_max, 'a') as file:
        file.write(f'\nLayer {layer_id}: {maxes.flatten().tolist()}\n')
    EXP_ITERATIONS = 7
    x_exp = approx_exp(x-maxes, EXP_ITERATIONS)
    # x_exp = torch.exp(x-maxes)
    x_exp[x <= -3.4028e+37] = 0
    with open(output_max, 'a') as file:
        file.write(f'\nExp {layer_id}: {x_exp.flatten().tolist()}\n')
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
    G_ITERATIONS = 7
    if torch.cuda.is_available():
        normalizer = normalizer.to('cuda')
        # print(f"Device: {normalizer.device}")

    out = approx_div(x_exp / normalizer, x_exp_sum / normalizer,
                     G_ITERATIONS)

    # Useful for handpicking initial approx.
    # print((1/torch.mean(x_exp, dim, keepdim=True)).mean())
    return out

nan_logged = False

def approx_softmax(x, layer_id, dim=None): #FINAL FUNCTION WITH ALL APPROXIMATIONS
    global nan_logged
    correct_maxes = torch.max(x, dim, keepdim=True)[0]
    # assert(correct_maxes == maxes)
    maxes = tensors[layer_id]
    x = x.double()
    maxes = maxes.double()
    EXP_ITERATIONS = 7
    x_diff = (x - maxes).clamp(min=-100, max=100)  # Prevent extreme negatives
    x_exp = approx_exp(x_diff, EXP_ITERATIONS)
    # x_exp = torch.exp(x-maxes)
    if not nan_logged and torch.isnan(x_exp).any():
        with open("nan_debug_log.txt", "a") as log_file:
            log_file.write(f"NaN detected in x_exp for layer_id: {layer_id}\n")
            log_file.write(f"Input x: {x.tolist()}\n")
            log_file.write(f"Input x shape: {x.shape}\n")
            log_file.write(f"x_diff: {x_diff.tolist()}\n")
            log_file.write(f"x_exp: {x_exp.tolist()}\n\n")
        nan_logged = True

    x_exp[x <= -3.4028e+37] = 0
    # assert torch.all(x_exp <= 1)
    # x_exp = torch.exp(x-maxes)

    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    if torch.any(x_exp_sum == 0):  # Avoid division by zero
        x_exp_sum += 1e-8
    if not nan_logged and torch.isnan(x_exp_sum).any():
        with open("nan_debug_log.txt", "a") as log_file:
            log_file.write(f"NaN detected in x_exp_sum for layer_id: {layer_id}\n")
            log_file.write(f"Input x: {x.tolist()}\n")
            log_file.write(f"Input x shape: {x.shape}\n")
            log_file.write(f"x_diff: {x_diff.tolist()}\n")
            log_file.write(f"x_exp: {x_exp.tolist()}\n")
            log_file.write(f"x_exp_sum: {x_exp_sum.tolist()}\n\n")
        nan_logged = True

    # return x_exp/x_exp_sum

    # Division
    # out = x_exp/x_exp_sum
    normalizer = torch.ones(x_exp.shape).sum(dim, keepdim=True)

    # assert torch.all(x_exp_sum / normalizer <= 1)

    # norm: divide by length so that quotient is <1 (denominator
    # becomes the mean)
    G_ITERATIONS = 7
    if torch.cuda.is_available():
        normalizer = normalizer.to('cuda')
        # print(f"Device: {normalizer.device}")

    out = approx_div(x_exp / normalizer, x_exp_sum / normalizer,
                     G_ITERATIONS)
    if not nan_logged and torch.isnan(out).any():
        with open("nan_debug_log.txt", "a") as log_file:
            log_file.write(f"NaN detected in out for layer_id: {layer_id}\n")
            log_file.write(f"Input x shape: {x.shape}\n")
            log_file.write(f"x - maxes: {x_diff.tolist()}\n")
            log_file.write(f"x_exp: {x_exp.tolist()}\n")
            log_file.write(f"x_exp_sum: {x_exp_sum.tolist()}\n")
            log_file.write(f"Output out: {out.tolist()}\n\n")
        nan_logged = True

    # Useful for handpicking initial approx.
    # print((1/torch.mean(x_exp, dim, keepdim=True)).mean())
    return out
