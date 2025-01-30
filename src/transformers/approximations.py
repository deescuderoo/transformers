from math import ceil, tanh, log2, sqrt, pi
import scipy.special
import torch, os, ast
import numpy as np
import json
from max_tensors.gpt2_medium_maxes import tensors_gpt2_medium_piqa, tensors_gpt2_medium_race
from max_tensors.gpt2_maxes import tensors_gpt2_piqa, tensors_gpt2_wsc, tensors_gpt2_arc, max_constants_gpt2_arc, max_constants_gpt2_piqa
from max_tensors.gpt2_large_maxes import tensors_gpt2_large_piqa

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

# Initialize the global variable
current_cycle = 0  # Default value is now 0

# Global variables
OUTPUT_FOLDER = "output_cycles"
LAYER_MAX_VALUES_FILE = os.path.join(OUTPUT_FOLDER, "layer_max_values.txt")
SHAPE_MISMATCH_FILE = os.path.join(OUTPUT_FOLDER, "shape_mismatch_log.txt")

# Ensure the folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize the max values file if not already present
if not os.path.exists(LAYER_MAX_VALUES_FILE):
    with open(LAYER_MAX_VALUES_FILE, 'w') as file:
        for layer_id in range(12):  # Assuming 12 layers
            file.write(f"Layer {layer_id}: {[-10000] * 12}\n")  # Use -10000

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

# layer0_array=[1.8493261337280273, 10.321573257446289, 1.0337244272232056, 11.179471969604492, 6.565133094787598, 16.084802627563477, 3.021239757537842, 9.312780380249023, 10.720375061035156, 12.07174301147461, 10.763772964477539, 10.797805786132812]
# layer1_array=[4.33711051940918, 3.385294198989868, 1.053083896636963, 2.346316337585449, 0.9965879321098328, 4.807539463043213, 4.680234909057617, 1.9316093921661377, 2.231010913848877, 2.1602818965911865, 2.092066764831543, 8.487981796264648]
# layer2_array=[1.7087633609771729, 2.9535627365112305, 4.604798793792725, 2.7798538208007812, 3.2145142555236816, 3.3662657737731934, 2.259106159210205, 2.492807388305664, 3.4144763946533203, 2.8978257179260254, 2.2755203247070312, 2.1004021167755127]
# layer3_array=[7.631556510925293, 7.5165534019470215, 6.794878005981445, 6.580643177032471, 6.48328971862793, 6.706857681274414, 6.219317436218262, 6.2063679695129395, 6.263098239898682, 5.790459632873535, 6.246354579925537, 6.452132225036621]
# layer4_array=[3.4940876960754395, 1.149423599243164, 1.2998638153076172, 2.6921796798706055, 2.8489766120910645, 0.77414870262146, 0.46880295872688293, 4.821672439575195, 2.983339309692383, 1.9948464632034302, 2.348292112350464, 39.45146179199219]
# layer5_array=[5.923620223999023, 11.5446195602417, 2.7273964881896973, 2.7468600273132324, 2.67863130569458, 7.868797302246094, 4.552934169769287, 4.86293888092041, 5.042206764221191, 5.317700386047363, 4.951260089874268, 5.159172058105469]
# layer6_array=[2.7284209728240967, 2.200655460357666, 1.024259328842163, 1.5481042861938477, 3.5086355209350586, 0.7202161550521851, 2.993629217147827, 0.934133768081665, 2.356409788131714, 9.204235076904297, 4.28968620300293, 1.8297182321548462]
# layer7_array=[1.9839451313018799, 6.594324588775635, 13.034424781799316, 1.9016940593719482, 2.6689841747283936, 2.9748497009277344, 2.595259666442871, 8.867525100708008, 4.641799449920654, 4.675210475921631, 11.67599868774414, 10.335289001464844]
# layer8_array=[2.3718461990356445, 10.817716598510742, 4.544055938720703, 4.658476829528809, 2.6210646629333496, 1.6930010318756104, 6.571752071380615, 4.000815391540527, 4.905542373657227, 5.262595176696777, 5.410916805267334, 5.95646858215332]
# layer9_array=[5.831937313079834, 6.704681396484375, 4.681728363037109, 3.184159278869629, 4.5470428466796875, 3.4491376876831055, 6.640199661254883, 3.4697933197021484, 4.080732822418213, 7.8267340660095215, 4.844199180603027, 5.132279396057129]
# layer10_array=[4.52237606048584, 7.927225112915039, 6.176588535308838, 6.088727951049805, 5.833307266235352, 4.04454231262207, 5.550886631011963, 6.582158088684082, 5.128731727600098, 5.044299125671387, 5.436607837677002, 7.902202606201172]
# layer11_array=[4.992722511291504, 5.467639923095703, 5.714563846588135, 4.192873001098633, 4.425979137420654, 6.185069561004639, 5.388833045959473, 6.145012855529785, 19.679534912109375, 7.636826515197754, 7.3761444091796875, 43.82441329956055]

layer0_array=[0.2962226867675781, 7.685041427612305, -0.024182945489883423, 7.261965751647949, 5.690557479858398, 10.67147445678711, 0.05856044590473175, 1.3883235454559326, 0.5720242261886597, -0.4859350919723511, 2.302060127258301, -0.5009303092956543]
layer1_array=[3.9619102478027344, 2.881669521331787, 0.09765380620956421, 0.7769048810005188, 0.7627350091934204, 4.5555949211120605, 3.775596857070923, 0.9124335050582886, 2.0618038177490234, 2.1276638507843018, 0.13773870468139648, 6.793796539306641]
layer2_array=[1.4515208005905151, 2.476414918899536, 3.649444103240967, 2.948371410369873, 1.498098611831665, 2.8711094856262207, 1.4750465154647827, 0.7527820467948914, 2.880405902862549, 2.0265679359436035, 1.1257076263427734, 1.2808010578155518]
layer3_array=[5.901467323303223, -0.4758477210998535, 0.5475810766220093, -1.3662619590759277, 2.031717300415039, 0.5066350698471069, -0.17573833465576172, -2.3956589698791504, 0.17406737804412842, 0.46715641021728516, -0.17265403270721436, -0.09717702865600586]
layer4_array=[2.043872356414795, 0.32703566551208496, 0.7177494764328003, 1.5054121017456055, 1.9527418613433838, 1.1387403011322021, -0.9419839382171631, 3.6026644706726074, 2.589608669281006, 0.5328593254089355, 1.833370327949524, 29.308387756347656]
layer5_array=[5.5738677978515625, 9.21202278137207, 3.222045421600342, 0.7668744921684265, 1.3584561347961426, 6.071133613586426, 2.8840153217315674, 2.5840015411376953, 4.390895366668701, 1.4839723110198975, 3.1228458881378174, 2.157837390899658]
layer6_array=[1.2103257179260254, 1.499786376953125, 0.8996633887290955, 0.9384918212890625, 2.205781936645508, -0.11252517998218536, 2.5786972045898438, 0.4000643491744995, 3.3342833518981934, 6.103588104248047, 3.7864296436309814, 1.1131362915039062]
layer7_array=[1.4054447412490845, 5.668951988220215, 10.570466995239258, 1.1217714548110962, 1.9667186737060547, 2.0836734771728516, 1.820041298866272, 7.9112443923950195, 2.0619773864746094, 2.0622212886810303, 10.240470886230469, 8.94202709197998]
layer8_array=[1.9705792665481567, 8.281548500061035, 3.9470791816711426, 3.800917625427246, 1.651401400566101, 1.7500579357147217, 4.530993461608887, 1.9340118169784546, 3.066556692123413, 2.105173110961914, 1.953851342201233, 2.246424436569214]
layer9_array=[3.8728508949279785, 5.830446720123291, 4.3037109375, 1.9596188068389893, 4.151242256164551, 2.36012601852417, 5.0724382400512695, 1.77933669090271, 3.2911128997802734, 5.991390705108643, 2.4027817249298096, 4.669929504394531]
layer10_array=[4.251124382019043, 5.736339569091797, 4.583291530609131, 4.907354354858398, 5.426852226257324, 2.907595634460449, 4.639364242553711, 4.683046340942383, 4.354070663452148, 3.020540237426758, 4.567971229553223, 7.825965881347656]
layer11_array=[5.02396297454834, 4.904455661773682, 5.005637168884277, 3.7745230197906494, 3.574686050415039, 5.328423500061035, 4.664555072784424, 6.03653621673584, 16.65554428100586, 5.48862361907959, 7.034262657165527, 43.84136199951172]
#
# device = torch.device("cuda")
#
# tensors = {}
# tensors[0] = torch.tensor(layer0_array, device=device).reshape(1, 12, 1, 1)
# tensors[1] = torch.tensor(layer1_array, device=device).reshape(1, 12, 1, 1)
# tensors[2] = torch.tensor(layer2_array, device=device).reshape(1, 12, 1, 1)
# tensors[3] = torch.tensor(layer3_array, device=device).reshape(1, 12, 1, 1)
# tensors[4] = torch.tensor(layer4_array, device=device).reshape(1, 12, 1, 1)
# tensors[5] = torch.tensor(layer5_array, device=device).reshape(1, 12, 1, 1)
# tensors[6] = torch.tensor(layer6_array, device=device).reshape(1, 12, 1, 1)
# tensors[7] = torch.tensor(layer7_array, device=device).reshape(1, 12, 1, 1)
# tensors[8] = torch.tensor(layer8_array, device=device).reshape(1, 12, 1, 1)
# tensors[9] = torch.tensor(layer9_array, device=device).reshape(1, 12, 1, 1)
# tensors[10] = torch.tensor(layer10_array, device=device).reshape(1, 12, 1, 1)
# tensors[11] = torch.tensor(layer11_array, device=device).reshape(1, 12, 1, 1)

# mean_constants=[]
# mean_constants.append(np.mean(layer0_piqa))
# mean_constants.append(np.mean(layer1_piqa))
# mean_constants.append(np.mean(layer2_piqa))
# mean_constants.append(np.mean(layer3_piqa))
# mean_constants.append(np.mean(layer4_piqa))
# mean_constants.append(np.mean(layer5_piqa))
# mean_constants.append(np.mean(layer6_piqa))
# mean_constants.append(np.mean(layer7_piqa))
# mean_constants.append(np.mean(layer8_piqa))
# mean_constants.append(np.mean(layer9_piqa))
# mean_constants.append(np.mean(layer10_piqa))
# mean_constants.append(np.mean(layer11_piqa))
#
# max_constants=[]
# max_constants.append(np.max(layer0_piqa))
# max_constants.append(np.max(layer1_piqa))
# max_constants.append(np.max(layer2_piqa))
# max_constants.append(np.max(layer3_piqa))
# max_constants.append(np.max(layer4_piqa))
# max_constants.append(np.max(layer5_piqa))
# max_constants.append(np.max(layer6_piqa))
# max_constants.append(np.max(layer7_piqa))
# max_constants.append(np.max(layer8_piqa))
# max_constants.append(np.max(layer9_piqa))
# max_constants.append(np.max(layer10_piqa))
# max_constants.append(np.max(layer11_piqa))
testsuite = "piqa"

def ref_softmax(x, dim=None):
    # x[x <= -3.4028e+37] = 0
    # print("input shape", x.shape)
    maxes = torch.max(x, dim, keepdim=True)[0]
    x_exp = torch.exp(x-maxes)
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    return x_exp/x_exp_sum

MAX_VALUES_FILE = "layer_max_tensors_gpt2_medium_arc.txt"

# Global file path for fallback logs
FALLBACK_LOG_FILE = "fallback_counts_log.txt"

# Initialize fallback log file
with open(FALLBACK_LOG_FILE, 'w') as file:
    file.write("Fallback counts per layer:\n")
    for layer_id in range(12):  # Assuming 12 layers
        file.write(f"Layer {layer_id}: 0 fallbacks\n")

# Global counter to track fallback usage per layer
fallback_counter = [0] * 12  # Assuming 12 layers per input

# Ensure the folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def convert_model_name(model_name):
    return model_name.replace("-", "_")

def approx_softmax_store_in_file(x, layer_id, model, dim=None):
    # Get max values along the specified dimension
    maxes = torch.max(x, dim, keepdim=True)[0]  # Shape: [1, 12, W, 1]
    maxes_reshaped = maxes.squeeze(0).squeeze(-1).cpu().numpy()  # Shape: [12, W]
    MAX_VALUES_FILE = f"layer_max_tensors_{model}_arc.txt"
    # Check if the file exists and if the layer's data is already present
    if not os.path.exists(MAX_VALUES_FILE):
        with open(MAX_VALUES_FILE, 'w') as file:
            file.write("")  # Create an empty file if it doesn't exist

    with open(MAX_VALUES_FILE, 'r+') as file:
        lines = file.readlines()

        # Check if the layer already has data
        layer_header = f"Layer {layer_id}:\n"
        if layer_header in lines:
            # Layer exists; update its data
            start_idx = lines.index(layer_header) + 1
            size_idx = start_idx
            tensor_idx = start_idx + 1

            # Load the current max tensor from the file
            current_size = eval(lines[size_idx].split(":")[1].strip())  # Parse size
            current_max_tensor = np.array(eval(lines[tensor_idx].strip()))  # Parse tensor

            # Handle size mismatch by resizing tensors
            updated_size = (max(current_size[0], maxes_reshaped.shape[0]),  # Rows
                            max(current_size[1], maxes_reshaped.shape[1]))  # Columns

            # Resize the current max tensor to match the updated size
            resized_current_max_tensor = np.full(updated_size, float('-inf'))  # Initialize with -inf
            resized_current_max_tensor[:current_size[0], :current_size[1]] = current_max_tensor

            # Resize the new tensor to match the updated size
            resized_maxes_reshaped = np.full(updated_size, float('-inf'))  # Initialize with -inf
            resized_maxes_reshaped[:maxes_reshaped.shape[0], :maxes_reshaped.shape[1]] = maxes_reshaped

            # Update the max tensor element-wise
            updated_max_tensor = np.maximum(resized_current_max_tensor, resized_maxes_reshaped)

            # Update the file content
            lines[size_idx] = f"Size: {updated_size}\n"
            lines[tensor_idx] = f"{updated_max_tensor.tolist()}\n"
        else:
            # Layer doesn't exist; write new data
            updated_size = maxes_reshaped.shape  # Get the current size
            lines.extend([
                f"{layer_header}",
                f"Size: {updated_size}\n",
                f"{maxes_reshaped.tolist()}\n\n"
            ])

        # Write the updated content back to the file
        file.seek(0)
        file.writelines(lines)

    # Perform the rest of the softmax approximation as usual
    EXP_ITERATIONS = 7
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
    G_ITERATIONS = 7
    if torch.cuda.is_available():
        normalizer = normalizer.to('cuda')
        # print(f"Device: {normalizer.device}")

    out = approx_div(x_exp / normalizer, x_exp_sum / normalizer,
                     G_ITERATIONS)

    # Useful for handpicking initial approx.
    # print((1/torch.mean(x_exp, dim, keepdim=True)).mean())
    return out

def approx_softmax_without_max_replacement(x, layer_id, model, dim=None): #FINAL FUNCTION WITHOUT MAX REPLACEMENT
    # correct_maxes = torch.max(x, dim, keepdim=True)[0]
    # assert(correct_maxes == maxes)
    maxes = torch.max(x, dim, keepdim=True)[0]

    EXP_ITERATIONS = 7
    x_diff = (x - maxes).clamp(min=-100, max=100)  # Prevent extreme negatives
    x_exp = approx_exp(x_diff, EXP_ITERATIONS)
    # x_exp = torch.exp(x-maxes)

    x_exp[x <= -3.4028e+37] = 0
    # assert torch.all(x_exp <= 1)
    # x_exp = torch.exp(x-maxes)
    # x_exp = x_exp * mask

    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    x_exp_sum = torch.clamp(x_exp_sum, min=1e-12)

    # return x_exp/x_exp_sum
    # x_exp_sum = x_exp_sum * mask

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
    # out = out * mask  # Final masking for valid values
    # Useful for handpicking initial approx.
    # print((1/torch.mean(x_exp, dim, keepdim=True)).mean())
    return out

def approx_softmax(x, layer_id, model, dim=None): #FINAL FUNCTION WITH ALL APPROXIMATIONS
    # correct_maxes = torch.max(x, dim, keepdim=True)[0]
    # assert(correct_maxes == maxes)
    # maxes = tensors5[layer_id]
    # maxes = maxes[:, :, :x.shape[2], :]  #wsc best accuracy
    global testsuite
    model_name = convert_model_name(model)
    tensor_name = f"max_constants_{model_name}_{testsuite}"

    # Access the tensor dictionary dynamically
    tensor_dict = globals().get(tensor_name)
    maxes = torch.full((1,x.shape[1],x.shape[2],1),tensor_dict[layer_id])
    #maxes = tensor_dict[layer_id]
    #maxes = maxes[:, :, :x.shape[2], :]
    if torch.cuda.is_available():
             maxes = maxes.to('cuda')

    EXP_ITERATIONS = 7
    x_diff = (x - maxes).clamp(min=-100, max=100)  # Prevent extreme negatives
    x_exp = approx_exp(x_diff, EXP_ITERATIONS)
    # x_exp = torch.exp(x-maxes)

    x_exp[x <= -3.4028e+37] = 0
    # assert torch.all(x_exp <= 1)
    # x_exp = torch.exp(x-maxes)
    # x_exp = x_exp * mask

    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    x_exp_sum = torch.clamp(x_exp_sum, min=1e-12)

    # return x_exp/x_exp_sum
    # x_exp_sum = x_exp_sum * mask

    # Division
    # out = x_exp/x_exp_sum
    normalizer = torch.ones(x_exp.shape).sum(dim, keepdim=True)

    # assert torch.all(x_exp_sum / normalizer <= 1)

    # norm: divide by length so that quotient is <1 (denominator
    # becomes the mean)
    G_ITERATIONS = 7 #check
    if torch.cuda.is_available():
        normalizer = normalizer.to('cuda')
        # print(f"Device: {normalizer.device}")

    out = approx_div(x_exp / normalizer, x_exp_sum / normalizer,
                     G_ITERATIONS)
    # out = out * mask  # Final masking for valid values
    # Useful for handpicking initial approx.
    # print((1/torch.mean(x_exp, dim, keepdim=True)).mean())
    return out