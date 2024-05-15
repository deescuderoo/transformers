from transformers import GPT2Config, GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline, set_seed
from transformers import activations
import torch
from torch import nn
import numpy as np
from transformers import GPT2LMHeadModelNew


# Initializing a GPT2 configuration

configuration = GPT2Config()

# This is the default GPT2 model from HF
std_model = GPT2LMHeadModel.from_pretrained('gpt2')

'''
The GPT2LMHeadModel class is a subclass of GPT2 that includes a
language modeling head on top of the base GPT-2 model. This language
modeling head is essentially a linear layer that takes the output of
the base model and projects it to the vocabulary size, allowing the
model to generate probability distributions over the vocabulary for
the next token in the sequence.
'''

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# CHANGING ACTIVATIONS
# This is the easiest one: one can instantiate a pretrained model and
# ask the loader to change the activation function. The "gelu_puma"
# name was added to the file
# ./src/transformers/activations.py See that file for details

new_config = GPT2Config.from_pretrained("gpt2", activation_function="gelu_puma")
puma_model = GPT2LMHeadModel.from_pretrained("gpt2", config=new_config)

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
    NEWTON_ITERATIONS = 20
    # Initial estimate
    y = initial_inv_sqrt(x)
    # Iterations
    for _ in range(NEWTON_ITERATIONS):
        y = (y * (3 - x * y**2)) / 2
    return y


def ref_inv_sqrt(x):
    '''
    Reference implementation of 1/sqrt(x) for comparison
    '''
    return 1/np.sqrt(x)


class NewLayerNorm(nn.Module):
    def __init__(self, old_ln):
        super().__init__()
        self.weights = old_ln.weight
        self.bias = old_ln.bias
        self.eps = old_ln.eps

    def forward(self, x):
        # Scales the variance down by SCALE_ROOT^2. Important to fit
        # in the required range
        SCALE_ROOT = 10

        length = x.shape[-1]
        mean = x.mean(-1, keepdim=True)

        diff = x - mean
        var = (diff**2).sum(-1, keepdim=True) / length
        sqrt_input = (var + self.eps) / SCALE_ROOT**2

        newton = newton_inv_sqrt(sqrt_input)
        ref = ref_inv_sqrt(sqrt_input)

        y = diff * (newton) * self.weights / SCALE_ROOT + self.bias

        return y


# Now, we modify the blocks in the transformer. With
# model.transfomer.h we access the hidden architecture,
# which is a ModuleList with the two layers that comprise
# the feed forward block of the architecture.

ref_model = GPT2LMHeadModel.from_pretrained("gpt2", config=new_config)
new_model = GPT2LMHeadModel.from_pretrained("gpt2", config=new_config)

for block in ref_model.transformer.h:
    block.ln_1 = RefLayerNorm(block.ln_1)
    block.ln_2 = RefLayerNorm(block.ln_2)

for block in new_model.transformer.h:
    block.ln_1 = NewLayerNorm(block.ln_1)
    block.ln_2 = NewLayerNorm(block.ln_2)


sm_model = GPT2LMHeadModelNew.from_pretrained("gpt2", config=new_config)

alm_model = GPT2LMHeadModelNew.from_pretrained("gpt2", config=new_config)
for block in alm_model.transformer.h:
    block.ln_1 = NewLayerNorm(block.ln_1)
    block.ln_2 = NewLayerNorm(block.ln_2)


# std_model: standard model from HF
# puma_model: Puma using PyTorch's LN
# ref_model: Puma while keeping LN intact, except we use our reference
# implementation instead of PyTorch's
# new_model: Puma while changing LN to use the approximation
# sm_model: Puma, LN intact, approx softmax
# alm_model: Puma, LN modified, approx SM with exact max

# CHANGING SOFTMAX (TODO)


# TESTING A SENTENCE

# Disables gradient computation
std_model.eval()
puma_model.eval()
ref_model.eval()
new_model.eval()
sm_model.eval()
alm_model.eval()

prompt_text = "The secret for success is"

# Tokenize the prompt text
input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

# Generate and decode text
std_output = std_model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
std_generated_text = tokenizer.decode(std_output[0], skip_special_tokens=True)

sm_output = sm_model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
sm_generated_text = tokenizer.decode(sm_output[0], skip_special_tokens=True)

puma_output = puma_model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
puma_generated_text = tokenizer.decode(puma_output[0], skip_special_tokens=True)

alm_output = alm_model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
alm_generated_text = tokenizer.decode(alm_output[0], skip_special_tokens=True)

ref_output = ref_model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
ref_generated_text = tokenizer.decode(ref_output[0], skip_special_tokens=True)

new_output = new_model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
new_generated_text = tokenizer.decode(new_output[0], skip_special_tokens=True)



# The below was useful for finding the range of LN:
# range_end = np.percentile(RefLayerNorm.max_list, 90)
# range_start = np.percentile(RefLayerNorm.min_list, 10)


# Save the model
alm_model.save_pretrained("./gpt2-custom")
tokenizer.save_pretrained("./gpt2-custom")


