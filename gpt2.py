from transformers import GPT2Config, GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline, set_seed
from transformers import activations
import torch, os
from torch import nn
import numpy as np
from transformers import GPT2LMHeadModelNew
from layer_norm_tensors.layer_norm_arc_easy import tensors_gpt2_norm

# Several models are defined in this file:

# std_model: standard model from HF

# gelu_model: std only approximating the GeLU

# refln_model: GeLU is exact, and LN is exact too but it is
# implemented from scratch instead of using pytorch's LN (this was
# useful as a stepping stone)

# gelu_aprxln_model: same as before, but the layer normalization is
# approximated. Note that softmax is still the standard one in all
# these models so far

# gelu_stdln_aprxsm_model: approximate GeLU and softmax, but keep
# LayerNorm intact

# mod_model: this is the completely modified model, it approximates
# GeLU, LayerNorm and SoftMax


# Initializing a GPT2 configuration

configuration = GPT2Config()

gpt2 = "gpt2" # "gpt2-xl" "gpt2" "gpt-large" "gpt2-medium"

# This is the default GPT2 model from HF
std_model = GPT2LMHeadModel.from_pretrained(gpt2)

'''
The GPT2LMHeadModel class is a subclass of GPT2 that includes a
language modeling head on top of the base GPT-2 model. This language
modeling head is essentially a linear layer that takes the output of
the base model and projects it to the vocabulary size, allowing the
model to generate probability distributions over the vocabulary for
the next token in the sequence.
'''

tokenizer = GPT2Tokenizer.from_pretrained(gpt2)

# CHANGING ACTIVATIONS
# This is the easiest one: one can instantiate a pretrained model and
# ask the loader to change the activation function. The "gelu_puma"
# name was added to the file
# ./src/transformers/activations.py See that file for details

new_config = GPT2Config.from_pretrained(gpt2, activation_function="gelu_puma")
gelu_model = GPT2LMHeadModel.from_pretrained(gpt2, config=new_config)

# CHANGING LAYER NORMALIZATION

# First, we define the new layer normalization as a derived class from
# nn.Module


class RefLayerNorm(nn.Module):
    '''
    Mimics the LayerNorm from PyTorch exactly. This was useful as a
    starting point for the approximation
    '''
    # Statics, used for finding ranges
    min_list = []
    max_list = []

    def __init__(self, old_ln):
        super().__init__()
        self.weights = old_ln.weight
        self.bias = old_ln.bias
        self.eps = old_ln.eps

    def forward(self, x):
        length = x.shape[-1]
        mean = x.mean(-1, keepdim=True)

        # This is the ref (with the help of torch)
        # https://stackoverflow.com/questions/59830168/layer-normalization-in-pytorch
        # var = x.var(-1, keepdim = True, correction=0)
        # sqrt_input = var + self.eps
        # y = (x-mean)/torch.sqrt(sqrt_input) * self.weights + self.bias


        # This is the ref (manually)
        diff = x - mean
        var = (diff**2).sum(-1, keepdim=True) / length
        sqrt_input = var + self.eps
        y = (diff / torch.sqrt(sqrt_input)) * self.weights + self.bias

        # This is the rewrite from the non-interactive paper
        # z = length * (x - mean)
        # sqrt_input = (z**2).sum(-1, keepdim = True) + self.eps * length**3
        # y = np.sqrt(length) * (z / torch.sqrt(sqrt_input)) * self.weights + self.bias

        # Useful for finding ranges
        RefLayerNorm.min_list.append(sqrt_input.min())
        RefLayerNorm.max_list.append(sqrt_input.max())
        return y


def initial_inv_sqrt(x):
    '''
    Using Taylor to find the starting point for Newton's approximation
    of 1/sqrt(x)
    '''
    # Ranges found empirically using RefLayerNorm
    RANGE_START = 0
    RANGE_END = 100
    center = (RANGE_START + RANGE_END)/2 + 1
    sq = np.sqrt(center)
    y = 1/sq - (x-center) / (2 * sq**3) \
        + (3 * (x-center)**2) / (8 * sq**5) \
        - (5 * (x-center)**3) / (16 * sq**7)
    return y


def newton_inv_sqrt(x):
    '''
    Newton approximation for 1/sqrt(x)
    '''
    NEWTON_ITERATIONS = 18 #check
    # Initial estimate
    y = initial_inv_sqrt(x)
    # Iterations
    for _ in range(NEWTON_ITERATIONS):
        y = (y * (3 - x * y**2)) / 2
    return y
    # return 1/torch.sqrt(x)


def ref_inv_sqrt(x):
    '''
    Reference implementation of 1/sqrt(x) for comparison
    '''
    return 1/torch.sqrt(x)

layernorm_replace_counter = 0

class NewLayerNorm(nn.Module):
    def __init__(self, old_ln):
        super().__init__()
        self.weights = old_ln.weight
        self.bias = old_ln.bias
        self.eps = old_ln.eps

    def forward(self, x):
        # print("running approx layernorm")
        # Scales the variance down by SCALE_ROOT^2. Important to fit
        # in the required range
        SCALE_ROOT = 60 #check
        #length = x.shape[-1]
        mean = x.mean(-1, keepdim=True)

        diff = x - mean
        #var = (diff**2).sum(-1, keepdim=True) / length
        #sqrt_input = (var + self.eps) / SCALE_ROOT**2
        #newton = newton_inv_sqrt(sqrt_input)
        global layernorm_replace_counter  # Use global counter
        if tensors_gpt2_norm[layernorm_replace_counter] is not None:
            norm_tensor = tensors_gpt2_norm[layernorm_replace_counter]
        else:
            raise ValueError(f"LayerNorm tensor at index {layernorm_replace_counter} is missing!")
        norm_tensor = norm_tensor[:, :x.shape[1], :]

        if torch.cuda.is_available():
            norm_tensor = norm_tensor.to('cuda')
        layernorm_replace_counter = (layernorm_replace_counter + 1) % 24
        # newton = torch.full([1,x.shape[1],1],14, device= "cuda")

        y = diff * (norm_tensor) * self.weights / SCALE_ROOT + self.bias

        return y

LAYERNORM_VALUES_FILE = "layernorm_max_tensors.txt"
layernorm_counter = 0  # Tracks LayerNorm calls (0 to 23)

# Check if the file exists
if not os.path.exists(LAYERNORM_VALUES_FILE):
    with open(LAYERNORM_VALUES_FILE, 'w') as file:
        file.write("")  # Create an empty file if it doesn't exist

def update_layernorm_max(layernorm_tensor):
    global layernorm_counter

    layer_id = (layernorm_counter // 2) + 1  # Layer number (1-based)
    norm_index = (layernorm_counter % 2) + 1  # 1 or 2
    layer_header = f"Layer {layer_id} - LayerNorm {norm_index}:\n"

    if isinstance(layernorm_tensor, torch.Tensor):
        layernorm_tensor = layernorm_tensor.detach().cpu().numpy()

    # Ensure the tensor is at least 2D (rows, cols)
    layernorm_tensor = layernorm_tensor.squeeze()  # Removes singleton dimensions
    if layernorm_tensor.ndim == 0:
        layernorm_tensor = np.array([[layernorm_tensor]])  # Convert scalar to (1,1)
    elif layernorm_tensor.ndim == 1:
        layernorm_tensor = np.expand_dims(layernorm_tensor, axis=1)  # Convert (N,) to (N,1)

    if np.isnan(layernorm_tensor).any():
        #print(f"Warning: NaNs detected in Layer {layer_id} - LayerNorm {norm_index}, replacing with -1e9")
        layernorm_tensor = np.nan_to_num(layernorm_tensor, nan=-1e9)

    # Ensure file exists
    if not os.path.exists(LAYERNORM_VALUES_FILE):
        with open(LAYERNORM_VALUES_FILE, 'w') as file:
            file.write("")

    with open(LAYERNORM_VALUES_FILE, 'r+') as file:
        lines = file.readlines()

        if layer_header in lines:
            start_idx = lines.index(layer_header) + 1
            size_idx = start_idx
            tensor_idx = start_idx + 1

            try:
                current_size = eval(lines[size_idx].split(":")[1].strip())
            except Exception as e:
                #print(f"Error parsing size for {layer_header.strip()}: {e}")
                layernorm_counter = (layernorm_counter + 1) % 24
                return

            try:
                current_max_tensor = np.array(eval(lines[tensor_idx].strip().replace("nan", "float('nan')")))
            except Exception as e:
                #print(f"Error parsing tensor for {layer_header.strip()}: {e}")
                layernorm_counter = (layernorm_counter + 1) % 24
                return

            updated_size = (max(current_size[0], layernorm_tensor.shape[0]),
                            max(current_size[1], layernorm_tensor.shape[1]))

            resized_current_max_tensor = np.full(updated_size, -1e9)
            resized_current_max_tensor[:current_size[0], :current_size[1]] = current_max_tensor

            resized_layernorm_tensor = np.full(updated_size, -1e9)
            resized_layernorm_tensor[:layernorm_tensor.shape[0], :layernorm_tensor.shape[1]] = layernorm_tensor

            updated_max_tensor = np.maximum(resized_current_max_tensor, resized_layernorm_tensor)

            lines[size_idx] = f"Size: {updated_size}\n"
            lines[tensor_idx] = f"{updated_max_tensor.tolist()}\n"
        else:
            updated_size = layernorm_tensor.shape
            lines.extend([
                f"{layer_header}",
                f"Size: {updated_size}\n",
                f"{layernorm_tensor.tolist()}\n\n"
            ])

        file.seek(0)
        file.writelines(lines)
        file.truncate()

    layernorm_counter = (layernorm_counter + 1) % 24

class NewLayerNormReplace(nn.Module):
    def __init__(self, old_ln):
        super().__init__()
        self.weights = old_ln.weight
        self.bias = old_ln.bias
        self.eps = old_ln.eps

    def forward(self, x):
        # print("running approx layernorm")
        # Scales the variance down by SCALE_ROOT^2. Important to fit
        # in the required range
        global global_max, global_min, global_sum, global_count, stats_file
        SCALE_ROOT = 30 #check

        length = x.shape[-1]
        mean = x.mean(-1, keepdim=True)

        diff = x - mean
        var = (diff**2).sum(-1, keepdim=True) / length
        sqrt_input = (var + self.eps) / SCALE_ROOT**2

        newton = newton_inv_sqrt(sqrt_input)
        # newton = ref_inv_sqrt(sqrt_input)
        update_layernorm_max(newton)

        y = diff * (newton) * self.weights / SCALE_ROOT + self.bias

        return y


# Now, we modify the blocks in the transformer. With
# model.transfomer.h we access the hidden architecture,
# which is a ModuleList with the two layers that comprise
# the feed forward block of the architecture.

refln_model = GPT2LMHeadModel.from_pretrained(gpt2)
gelu_aprxln_model = GPT2LMHeadModel.from_pretrained(gpt2, config=new_config)

#for block in refln_model.transformer.h:
#    block.ln_1 = RefLayerNorm(block.ln_1)
#    block.ln_2 = RefLayerNorm(block.ln_2)

#for block in gelu_aprxln_model.transformer.h:
#    block.ln_1 = NewLayerNorm(block.ln_1)
#    block.ln_2 = NewLayerNorm(block.ln_2)


gelu_stdln_aprxsm_model = GPT2LMHeadModelNew.from_pretrained(gpt2, config=new_config)

mod_model = GPT2LMHeadModelNew.from_pretrained(gpt2, config=new_config)
for block in mod_model.transformer.h:
    block.ln_1 = NewLayerNorm(block.ln_1)
    block.ln_2 = NewLayerNorm(block.ln_2)



####################################################################
# TESTING A SENTENCE

# Disables gradient computation
std_model.eval()
if torch.cuda.is_available(): std_model.to('cuda')

gelu_model.eval()
if torch.cuda.is_available(): gelu_model.to('cuda')

refln_model.eval()
if torch.cuda.is_available(): refln_model.to('cuda')

gelu_aprxln_model.eval()
if torch.cuda.is_available(): gelu_aprxln_model.to('cuda')

gelu_stdln_aprxsm_model.eval()
if torch.cuda.is_available(): gelu_stdln_aprxsm_model.to('cuda')

mod_model.eval()
if torch.cuda.is_available(): mod_model.to('cuda')


prompt_text = "The secret for success is"

### Tokenize the prompt text
input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
if torch.cuda.is_available(): input_ids = input_ids.to('cuda')


### Generate and decode text

std_output = std_model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
std_generated_text = tokenizer.decode(std_output[0], skip_special_tokens=True)
print("------------------------------------------\n")
print(f"std output:\n{std_generated_text}")

#gelu_output = gelu_model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
#gelu_generated_text = tokenizer.decode(gelu_output[0], skip_special_tokens=True)
#print("------------------------------------------\n")
#print(f"gelu output:\n{gelu_generated_text}")

#refln_output = refln_model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
#refln_generated_text = tokenizer.decode(refln_output[0], skip_special_tokens=True)
#print("------------------------------------------\n")
#print(f"refln output:\n{refln_generated_text}")

#gelu_aprxln_output = gelu_aprxln_model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
#gelu_aprxln_generated_text = tokenizer.decode(gelu_aprxln_output[0], skip_special_tokens=True)
#print("------------------------------------------\n")
#print(f"gelu_aprxln output:\n{gelu_aprxln_generated_text}")

#gelu_stdln_aprxsm_output = gelu_stdln_aprxsm_model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
#gelu_stdln_aprxsm_generated_text = tokenizer.decode(gelu_stdln_aprxsm_output[0], skip_special_tokens=True)
#print("------------------------------------------\n")
#print(f"gelu_stdln_aprxsm output:\n{gelu_stdln_aprxsm_generated_text}")

mod_output = mod_model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
mod_generated_text = tokenizer.decode(mod_output[0], skip_special_tokens=True)
print("------------------------------------------\n")
print(f"mod output:\n{mod_generated_text}")


####################################################################


# Save the model
# mod_model.save_pretrained("./gpt2-custom")
# tokenizer.save_pretrained("./gpt2-custom")


####################################################################

# LM EVAL using the evaluation harness

import lm_eval

from lm_eval.models.huggingface import HFLM
# Uncomment the desired tasks
tasks = [
    # "lambada_openai",
    # "hellaswag"
    "arc_easy"
    # "wikitext", -- not accurate
    # "glue",
    # "storycloze", -- needs custom dataset download
    # "wsc",
    # "lambada_cloze",
    # "lambada_standard_cloze_yaml" --nan,
    # "piqa"
    # "race"
    # "triviaqa"
        ]
batch_size = 8
task_manager = lm_eval.tasks.TaskManager()


# Modified model
mod_model_lmeval = mod_model
if torch.cuda.is_available(): mod_model_lmeval.to('cuda')
mod_model_lmeval = HFLM(pretrained=mod_model_lmeval)

mod_results = lm_eval.simple_evaluate( # call simple_evaluate
    model=mod_model_lmeval,
    tasks=tasks,
    num_fewshot=0,
    task_manager=task_manager,
    batch_size=batch_size)


# Standard model
std_model_lmeval = std_model
if torch.cuda.is_available(): std_model_lmeval.to('cuda')
std_model_lmeval = HFLM(pretrained=std_model_lmeval)

std_results = lm_eval.simple_evaluate( # call simple_evaluate
    model=std_model_lmeval,
    tasks=tasks,
    num_fewshot=0,
    task_manager=task_manager,
    batch_size=batch_size)


print("Modified:")
print(mod_results['results'])
#
print("Standard:")
print(std_results['results'])


# # Below: code for saving and loading the results 
# import pickle

# def save_dict(dictionary, name):
#     with open(name+'.pkl', 'wb') as f: pickle.dump(dictionary, f)

# # Utilities for loading dictionaries later on
# import os

# # Set the directory you want to work with
# directory = "."  # The current directory
# results = {}

# # Loop through each file in the directory
# for filename in os.listdir(directory):
#     # Construct the full file path
#     file_path = os.path.join(directory, filename)

# # Check if it's a file
# if os.path.isfile(file_path):
#     # Do something with the file
#     print(f"File: {filename}")
#     with open(filename, 'rb') as f:
#         results[filename] = pickle.load(f)['results']
#     # You can read the file contents, process the file, etc
