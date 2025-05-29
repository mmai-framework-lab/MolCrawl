from transformers import BertConfig, BertForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk

model_path = ""
max_length = 1024
dataset_dir = ""
learning_rate = 6e-6
weight_decay = 1e-1
warmup_steps = 200
max_steps = 600000
batch_size = 10

gradient_accumulation_steps = 5 * 8
per_device_eval_batch_size = 1
log_interval = 100
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open("bert/configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------


if not ("meta_vocab_size" in vars() and "meta_vocab_size" in globals()):
    try:
        meta_vocab_size = (len(tokenizer) // 8 + 1) * 8
    except Exception:
        raise ImportError(
            "Please initialize the variable meta_vocab_size in the *_config.py file with the size of your vocabulary."
        )

if model_size == "small":
    model_config = BertConfig(vocab_size=meta_vocab_size, max_position_embeddings=max_length)
elif model_size == "medium":
    # Note that this would be bert-large but the size is equivalent to gpt2-medium so we name it medium here as well
    model_config = BertConfig(
        vocab_size=meta_vocab_size,
        max_position_embeddings=max_length,
        hidden_size=1024,  # Dimensionality of the encoder layers and the pooler layer
        num_hidden_layers=24,  # Number of hidden layers in the Transformer encoder
        num_attention_heads=16,  # Number of attention heads
        intermediate_size=4096,  # Dimensionality of the "intermediate" (feed-forward) layer
    )
elif model_size == "large":
    # Custom config to match the size of gpt2 large model.
    model_config = BertConfig(
        vocab_size=meta_vocab_size,
        max_position_embeddings=max_length,
        hidden_size=1152,  # Hidden layer size
        num_hidden_layers=36,  # Number of transformer layers
        num_attention_heads=18,  # Number of attention heads
        intermediate_size=4608,  # Size of intermediate (feed-forward) layer
    )
else:
    raise ValueError("model_size: {model_size} is not supported choose between small, medium and large")

model = BertForMaskedLM(config=model_config)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.2)

training_args = TrainingArguments(
    output_dir=model_path,  # output directory to where save model checkpoint
    logging_strategy="steps",  # log every `logging_steps`
    logging_steps=log_interval,  # log every 1000 steps
    evaluation_strategy="steps",  # evaluate each `logging_steps` steps
    overwrite_output_dir=True,
    max_steps=max_steps,  # number of training epochs, feel free to tweak
    per_device_train_batch_size=batch_size,  # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=gradient_accumulation_steps,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=per_device_eval_batch_size,  # evaluation batch size
    save_steps=1000,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
)


train_dataset = load_from_disk(dataset_dir)["train"]
test_dataset = load_from_disk(dataset_dir)["test"].select(range(min(10000, load_from_disk(dataset_dir)["test"].num_rows)))  # for testing purposes, select only 10000 samples


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
