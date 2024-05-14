from transformers import GPT2Config, GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline, set_seed
from transformers import activations
import torch
from torch import nn
import numpy as np

# Initializing a GPT2 configuration

configuration = GPT2Config()

# Initializing a model

# model = GPT2Model(configuration)
# model = GPT2LMHeadModel(configuration)
model = GPT2LMHeadModel.from_pretrained('gpt2')
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
# /home/paz/Code/transformers/src/transformers/activations.py See that
# file for details

my_config = GPT2Config.from_pretrained("gpt2", activation_function="gelu_puma")
my_model = GPT2LMHeadModel.from_pretrained("gpt2", config=my_config)

# CHANGING LAYER NORMALIZATION

# First, we define the new layer normalization as a derived class from
# nn.Module

# def approx_ln(x):
#     return y

# Resources that helped me with the implementation of LN:
# https://stackoverflow.com/questions/59830168/layer-normalization-in-pytorch


def initial_inv_sqrt(x, interval_start, interval_end):
    center = (interval_start + interval_end)/2 + 1
    sq = np.sqrt(center)
    y = 1/sq - (x-center)/(2*sq**3) \
        + (3 * (x-center)**2) / (8*sq**5) \
        - (5 * (x-center)**3) / (16*sq**7)
    return y


def newton_inv_sqrt(x, iterations, interval_start, interval_end):
    y = initial_inv_sqrt(x, interval_start, interval_end)
    for i in range(iterations):
        y = (y * (3 - x * y**2)) / 2
    return y


class RefLayerNorm(nn.Module):
    def __init__(self, old_ln):
        super().__init__()
        self.weights = old_ln.weight
        self.bias = old_ln.bias
        self.eps = old_ln.eps

    def forward(self, x):
        length = x.shape[-1]
        mean = x.mean(-1, keepdim = True)
        diff = x - mean

        var = (diff**2).sum(-1, keepdim = True) / length
        y = (diff / torch.sqrt( var + self.eps )) * self.weights + self.bias
        return y


class NewLayerNorm(nn.Module):
    def __init__(self, old_ln):
        super().__init__()
        self.weights = old_ln.weight
        self.bias = old_ln.bias
        self.eps = old_ln.eps

    def forward(self, x):
        length = x.shape[-1]
        mean = x.mean(-1, keepdim = True)
        z = length * (x - mean)
        y = np.sqrt(length) * (z / torch.sqrt((z**2).sum(-1, keepdim = True) + self.eps * length**3 )) * self.weights + self.bias
        return y

# let us test w.r.t. torch's LN

batch_size = 2
seq_len = 5
hidden_size = 10
x = torch.randn(batch_size, seq_len, hidden_size)

LN = torch.nn.LayerNorm(normalized_shape=hidden_size)
newLN = NewLayerNorm(LN)
refLN = RefLayerNorm(LN)

ref_error = (refLN(x) - LN(x)).mean()
new_error = (newLN(x) - LN(x)).mean()
assert torch.allclose(refLN(x), LN(x))
assert torch.allclose(newLN(x), LN(x))

# Now, we modify the blocks in the transformer. With my_model.transfomer.h we access the hidden architecture, which is a ModuleList with the two layers that comprise the feed forward block of the architecture.

for block in my_model.transformer.h:
    print(block.ln_1)
    block.ln_1 = NewLayerNorm(block.ln_1)
    block.ln_2 = NewLayerNorm(block.ln_2)
# nn.LayerNorm(block.ln_1.normalized_shape)

# CHANGING SOFTMAX (TODO)



# Save the model

# my_model.save_pretrained("./gpt2-custom")
# tokenizer.save_pretrained("./gpt2-custom")
