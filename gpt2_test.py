from transformers import GPT2Config, GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline, set_seed
from transformers import activations

# Initializing a GPT2 configuration

configuration = GPT2Config()

# Initializing a model (with random weights) from the configuration

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


# Now we modify the config

# config = model.config
# new_relu = activations.PumaGELUActivation()

# for layer in model.transformer.h:
#     layer.set_activation_fn(new_relu)

my_config = GPT2Config.from_pretrained("gpt2", activation_function="gelu_puma")    
my_model = GPT2LMHeadModel.from_pretrained("gpt2", config=my_config)

text = "Create a poem."
input_ids = tokenizer(text, return_tensors='pt')
my_output = my_model.generate(
    **input_ids,
    max_length=100,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    early_stopping=True
)
output = model.generate(
    **input_ids,
    max_length=100,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    early_stopping=True
)
my_generated_text = tokenizer.decode(my_output[0], skip_special_tokens=True)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)


print(my_generated_text)
print(generated_text)

# Let's test!

from datasets import load_metric, load_dataset
from evaluate import TextClassificationEvaluator
from transformers import TextClassificationPipeline

# # dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
# dataset = load_dataset("imdb", split="test").shuffle().select(range(1000))

# text_classification_pipeline = TextClassificationPipeline(model=my_model, tokenizer=tokenizer)

# evaluator = TextClassificationEvaluator(
#     task_name="text-classification",
#     data_args=None,
#     metric_name="accuracy",
#     predictions=text_classification_pipeline(dataset["test"]["text"]),
#     label_list=dataset["test"]["labels"]
# )

# score = evaluator.evaluate()
# print(f"Accuracy: {score['accuracy']}")

my_model.save_pretrained("./gpt2-puma")
tokenizer.save_pretrained("./gpt2-puma")
