from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset, concatenate_datasets

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add a padding token to the tokenizer if it doesn't have one
tokenizer.pad_token = tokenizer.eos_token

# Load the public dataset from Hugging Face
public_dataset = load_dataset("conv_ai_2", trust_remote_code=True)  # Allow custom code to run

# Extract the relevant text data from the public dataset
def extract_dialogues(examples):
    # Extract the 'text' field from each dictionary in the 'dialog' list
    return {'text': [' '.join([utterance['text'] for utterance in dialogue]) for dialogue in examples['dialog']]}

public_dataset = public_dataset.map(extract_dialogues, batched=True, remove_columns=['id', 'dialog_id', 'bot_profile', 'user_profile', 'eval_score', 'profile_match'])

# Load your custom dataset
def load_custom_dataset(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    texts = [line.strip() for line in lines if line.strip()]
    custom_dataset = Dataset.from_dict({"text": texts})
    return custom_dataset

custom_dataset = load_custom_dataset("dataset.txt")

# Combine the public and custom datasets
combined_dataset = concatenate_datasets([public_dataset['train'], custom_dataset])

# Preprocess the combined dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

tokenized_dataset = combined_dataset.map(preprocess_function, batched=True, remove_columns=["text"])

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # No masked language modeling for GPT-2
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save the model
trainer.save_model("./fine-tuned-gpt2")
tokenizer.save_pretrained("./fine-tuned-gpt2")
