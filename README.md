# Modifying transformers for FHE + Benchmarks

## Files

The main files of this project are the following:

- [approximations.py](src/transformers/approximations.py). This file contains the approximations of the comparison function, max, exp, division and softmax.
- [activations.py](src/transformers/activations.py). This file is modified w.r.t. HuggingFace's. It contains the approximation of GeLU
- [modeling_gpt2_new_softmax.py](src/transformers/models/gpt2/modeling_gpt2_new_softmax.py). This file is a copy of GPT-2's implementation in HuggingFace ([modeling_gpt2.py](src/transformers/models/gpt2/modeling_gpt2.py), except that the SoftMax calls are replaced by our approximation
- [gpt2.py](gpt2.py). This is the main file and serves multiple purposes. It contains the approximation of inverse sqrt and LayerNorm. It defines multiple models, run them on a toy query, and performs the evaluation using the evaluation harness.

## Installation and Dependencies

Pull the repo and install the package as an editable package (preferably in an environment):
```
pip install -e .
```
Also, install the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) by
```
pip install lm-eval
```

## Running the `gpt2.py` file

Running `python gpt2.py` will execute the following.
First, several models are defined: 

- `std_model`: standard model from HF
- `gelu_model`: std only approximating the GeLU
- `refln_model`: GeLU is exact, and LN is exact too but it is implemented from scratch instead of using pytorch's LN (this was useful as a stepping stone)
- `gelu_aprxln_model`: same as before, but the layer normalization is approximated. Note that softmax is still the standard one in all these models so far
- `gelu_stdln_aprxsm_model`: approximate GeLU and softmax, but keep LayerNorm intact
- `mod_model`: this is the completely modified model, it approximates GeLU, LayerNorm and SoftMax

Then, a toy sentence "The secret for success is" is provided as inputs to all these variants, and the output is displayed on screen.
The final part of the file runs the LM evaluation harness to run the tasks `lambada_openai`, `hellaswag` and `arc_easy` on the models `std_model` and `mod_model`.
The results are displayed on screen.
