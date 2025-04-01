from datasets import load_dataset
from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import RobertaTokenizer
from datasets import DatasetDict, Dataset
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import pandas as pd

# dataset = load_dataset("google/code_x_glue_cc_code_to_code_trans")
# print("DATASET: ", type(dataset), dataset)

#should also add removing dataframe headers here - should also remove the row index numbers
train_dataset = pd.read_csv('../data/clean/ft_train.csv')
train_dataset = Dataset.from_pandas(train_dataset)
test_dataset = pd.read_csv('../data/clean/ft_test.csv')
test_dataset = Dataset.from_pandas(test_dataset)
valid_dataset = pd.read_csv('../data/clean/ft_valid.csv')
valid_dataset = Dataset.from_pandas(valid_dataset)
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': valid_dataset,
    'test': test_dataset
})
# train_methods = train_dataset['cleaned_method']
# train_target = train_dataset['target_block']
# print("TRAIN METHODS: ", train_methods)
# exit()

# print("Sample Python Code:", dataset['train'][0]['java'])
# print("Target Java Code:", dataset['train'][0]['cs'])

model_checkpoint = "Salesforce/codet5-small"

model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
#tokenizer.add_tokens(["<IF-STMT>"]) #Imagine we need an extra token. This line adds the extra token to the vocabulary
tokenizer.add_tokens(["<TAB>", "<MASK>"])

model.resize_token_embeddings(len(tokenizer))

#edited here

def preprocess_function(examples):
    # inputs = examples["java"]
    # targets = examples["cs"]
    inputs = examples['cleaned_method']
    targets = examples['target_block']
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# tokenized_datasets = dataset.map(preprocess_function, batched=True)
# exit()
tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./codet5-finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    logging_steps=100,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

metrics = trainer.evaluate(tokenized_datasets["test"])
print("Test Evaluation Metrics:", metrics)

input_code = "def add(a, b):\n    return a + b"
inputs = tokenizer(input_code, return_tensors="pt", padding=True, truncation=True)
outputs = model.generate(**inputs, max_length=256)
print("Generated Java Code:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))