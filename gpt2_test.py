from transformers import GPT2Config, GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline, set_seed
from transformers import activations
from torch import nn

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

class NewLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# Now, we modify the blocks in the transformer. With my_model.transfomer.h we access the hidden architecture, which is a ModuleList with the two layers that comprise the feed forward block of the architecture.

for block in my_model.transformer.h:
    print(block.ln_1)
    block.ln_1 = NewLayerNorm()
    block.ln_2 = NewLayerNorm()
# nn.LayerNorm(block.ln_1.normalized_shape)

# CHANGING SOFTMAX (TODO)



# Save the model

my_model.save_pretrained("./gpt2-custom")
tokenizer.save_pretrained("./gpt2-custom")
