from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import torch

# Load pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the pad token to the end-of-sequence token if not already set
tokenizer.pad_token = tokenizer.eos_token

# Example data for fine-tuning (replace with your actual dataset)
texts = [
    "This is a sample text for fine-tuning the GPT-2 model.",
    "Feel free to replace this with your actual data.",
    "The more diverse and comprehensive your data, the better your model will perform."
]

# Tokenize and encode the dataset
inputs = tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding=True)

# Define optimizer and learning rate schedule
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=100)

# Fine-tuning loop
model.train()
for epoch in range(3):  # You can adjust the number of epochs
    for i in range(len(inputs['input_ids'])):
        batch = inputs['input_ids'][i:i+1]  # Take one input at a time
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

# Save the fine-tuned model
model.save_pretrained('./fine-tuned-gpt2')
tokenizer.save_pretrained('./fine-tuned-gpt2')

print("Model fine-tuning complete and saved to './fine-tuned-gpt2'.")
