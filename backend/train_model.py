from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import torch

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Tokenize and encode the dataset
texts = ["Your dataset text here"]  # Replace with your actual dataset
inputs = tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding=True)

# Define the optimizer and learning rate schedule
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=100)

# Fine-tuning loop
model.train()
for epoch in range(3):  # Number of epochs
    for batch in inputs['input_ids']:
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

# Save the fine-tuned model
model.save_pretrained('../fine-tuned-gpt2')  # Save outside the backend folder
tokenizer.save_pretrained('../fine-tuned-gpt2')
