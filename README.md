# Modifying transformers for FHE + Benchmarks

The file [gpt2.py](gpt2.py) contains an example of how to modify different components of GPT2 (Softmax is still missing)

Pull the repo and install the package as an editable package (preferably in an environment):
```
pip install -e .
```

The current [gpt2.py](gpt2.py) stores a custom model in the directory `./gpt2-custom`. This is used by the evaluation library

# Evaluation

Evaluation is carried out by the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) library. We pass the argument `--model_args pretrained` and point to our local model. If we have a CUDA device, we can also specify it. 

```
lm_eval --model hf \
    --model_args pretrained="./gpt2-custom" \
    --tasks lambada_openai \
    --device cuda:0 \
    --batch_size 8
```

There are multiple benchmarks (aka "tasks") possible: `lm_eval --tasks list`.
