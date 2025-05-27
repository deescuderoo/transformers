from math import ceil, tanh, log2, sqrt, pi
import scipy.special
import torch, os, ast
import numpy as np
import json
from tqdm import tqdm
from .max_tensors.gpt2_medium_maxes import tensors_gpt2_medium_piqa, tensors_gpt2_medium_race
from .max_tensors.gpt2_maxes import tensors_gpt2_piqa, tensors_gpt2_wsc, tensors_gpt2_arc, max_constants_gpt2_arc, max_constants_gpt2_piqa, tensors_gpt2_race
from .max_tensors.gpt2_large_maxes import tensors_gpt2_large_piqa

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
    if torch.isnan(x).any():
        print(f"Warning: NaN detected in input tensor at layer {layer_id} for model {model}")

    # Get max values
    maxes = torch.max(x, dim, keepdim=True)[0]  # Shape: [1, 12, W, 1]
    maxes_reshaped = maxes.squeeze(0).squeeze(-1).cpu().numpy()  # Shape: [12, W]

    MAX_VALUES_FILE = f"layer_max_tensors_{model}_race.txt"

    # Check if the file exists
    if not os.path.exists(MAX_VALUES_FILE):
        with open(MAX_VALUES_FILE, 'w') as file:
            file.write("")  # Create an empty file if it doesn't exist

    with open(MAX_VALUES_FILE, 'r+') as file:
        lines = file.readlines()

        # Check if the layer already has data
        layer_header = f"Layer {layer_id}:\n"
        if layer_header in lines:
            start_idx = lines.index(layer_header) + 1
            size_idx = start_idx
            tensor_idx = start_idx + 1

            # Load size safely
            try:
                current_size = eval(lines[size_idx].split(":")[1].strip())
            except Exception as e:
                print(f"Error parsing size for layer {layer_id}: {e}")
                return

            # Load tensor safely
            try:
                current_max_tensor = np.array(eval(lines[tensor_idx].strip().replace("nan", "float('nan')")))
            except Exception as e:
                print(f"Error parsing tensor for layer {layer_id}: {e}")
                return

            # Handle size mismatch
            updated_size = (max(current_size[0], maxes_reshaped.shape[0]),
                            max(current_size[1], maxes_reshaped.shape[1]))

            # Initialize tensors safely
            resized_current_max_tensor = np.full(updated_size, -1e9)  # Large negative instead of -inf
            resized_current_max_tensor[:current_size[0], :current_size[1]] = current_max_tensor

            resized_maxes_reshaped = np.full(updated_size, -1e9)
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

        # Write back
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
    maxes = torch.max(x, dim, keepdim=True)[0]
    EXP_ITERATIONS = 7
    x_diff = (x - maxes).clamp(min=-100, max=100)  # Prevent extreme negatives
    x_exp = approx_exp(x_diff, EXP_ITERATIONS)

    x_exp[x <= -3.4028e+37] = 0

    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    x_exp_sum = torch.clamp(x_exp_sum, min=1e-12)

    normalizer = torch.ones(x_exp.shape).sum(dim, keepdim=True)
    G_ITERATIONS = 7
    if torch.cuda.is_available():
        normalizer = normalizer.to('cuda')

    out = approx_div(x_exp / normalizer, x_exp_sum / normalizer,
                     G_ITERATIONS)
    return out

def analyze_approx_softmax(x, layer_id, model, dim=None):
    filename="softmax_errors.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            results = json.load(f)
    else:
        results = {}
    gt = approx_softmax_without_max_replacement(x, layer_id, model, dim=-1)
    approx = approx_softmax(x, layer_id, model, dim=-1)
    results[f"layer_{layer_id}"] = {
        "MAE": (gt - approx).abs().mean().item(),
        "clipping_ratio": (approx == 0).float().mean().item(),
        "shape": list(x.shape)
    }
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    return approx

def approx_softmax(x, layer_id, model, dim=None): #FINAL FUNCTION WITH ALL APPROXIMATIONS
    global testsuite
    model_name = convert_model_name(model)
    tensor_name = f"tensors_{model_name}_{testsuite}"
    tensor_dict = globals().get(tensor_name)
    maxes = tensor_dict[layer_id]
    maxes = maxes[:, :, :x.shape[2], :] * 1.8
    if torch.cuda.is_available():
             maxes = maxes.to('cuda')
    x_diff = x- maxes
    EXP_ITERATIONS = 13
    x_exp = approx_exp(x_diff, EXP_ITERATIONS)
    x_exp[x <= -3.4028e+37] = 0
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    x_exp_sum = torch.clamp(x_exp_sum, min=1e-12)
    normalizer = torch.ones(x_exp.shape).sum(dim, keepdim=True)
    G_ITERATIONS = 22
    if torch.cuda.is_available():
        normalizer = normalizer.to('cuda')
    out = approx_div(x_exp / normalizer, x_exp_sum / normalizer,
                     G_ITERATIONS)
    return out