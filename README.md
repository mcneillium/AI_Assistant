# Fine-Tuned AI Assistant Using GPT-2

## Project Overview

This project aims to develop a customisable AI assistant by fine-tuning the GPT-2 model on a combination of public datasets (`conv_ai_2`) and custom datasets created for specific applications. The AI assistant is capable of responding to user queries in a contextually appropriate manner. The project demonstrates the process of fine-tuning a pre-trained language model and integrating it into various user interfaces, including web-based and desktop-based GUIs.

## Motivation

With the growing demand for specialised AI applications in customer support, education, and personal assistance, this project explores the creation of a tailored AI solution. By leveraging the power of GPT-2, the AI assistant can be adapted to different domains, providing valuable insights into the challenges and benefits of model fine-tuning.

## Project Setup

### Environment

- **Programming Language**: Python 3.11
- **Tools and Libraries**:
  - `transformers`: For model and tokenizer management.
  - `datasets`: For handling datasets from Hugging Face.
  - `torch`: For running and fine-tuning the GPT-2 model.
  - `flask`: For creating a web-based API.
  - `tkinter`: For creating a desktop GUI.
  - `streamlit`: For creating an interactive web application.
  - `boto3`: For interacting with AWS S3 (optional).

### Installation

To set up the environment:

1. **Create a Virtual Environment**:

   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate

2. **Install Dependencies**:

    pip install transformers datasets torch flask tkinter streamlit boto3

3. **Setting Up AWS (Optional): If using AWS for model storage, configure your AWS credentials**:

    aws configure

## Dataset Preparation

### Public Dataset

- **Dataset**: `conv_ai_2` from Hugging Face.  
  This dataset provides a collection of conversational data that the model can learn from. It contains dialogues between a user and a bot, covering various scenarios that help in fine-tuning the GPT-2 model for better conversational abilities.

### Custom Dataset

A custom dataset was also created to tailor the AI assistant to specific domains, such as technical support or customer service. This dataset consists of domain-specific dialogues and queries, formatted as a plain text file (`dataset.txt`). Each line in this file represents a single input-output pair or dialogue. This custom dataset was then combined with the public dataset to provide a richer and more contextually relevant training set.

## Model Fine-Tuning

### Process Overview

1. **Model and Tokenizer Loading**:

   from transformers import GPT2LMHeadModel, GPT2Tokenizer
   model = GPT2LMHeadModel.from_pretrained('gpt2')
   tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

2. **Dataset Combination**: The public dataset (conv_ai_2) was combined with the custom dataset to create a comprehensive training dataset.

3. **Tokenization**: The combined dataset was tokenized using the GPT-2 tokenizer. Special tokens, such as padding tokens, were added as needed:

    def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

4. **Training Configuration**: The training was configured with specific parameters, such as the number of epochs, batch size, and learning rate:

    training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

5. **Model Fine-Tuning**: The model was fine-tuned using the Trainer class, allowing for efficient training:

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
)
trainer.train()

6. **Model Saving**:
   After training, the fine-tuned model and tokenizer were saved for future deployment:

   trainer.save_model("./fine-tuned-gpt2")
   tokenizer.save_pretrained("./fine-tuned-gpt2")

## Reflection

### Challenges and Solutions

- **Dataset Preparation**: Combining public and custom datasets required careful preprocessing to ensure consistency in formatting and data structure.
- **Training Complexity**: Fine-tuning GPT-2 on large datasets was computationally intensive, but using a GPU significantly reduced training time and allowed for more iterations.

### What Went Well

- **Successful Integration**: The integration of the GPT-2 model with both web-based and desktop interfaces was smooth, allowing for flexible user interaction.
- **Model Performance**: The fine-tuned model generated contextually relevant responses, demonstrating the effectiveness of combining public and custom datasets for training.

### What Didn’t Go as Planned

- **Dataset Issues**: Initially, there were challenges in combining datasets with different structures, which required additional preprocessing steps.
- **Training Iterations**: The first few training configurations led to overfitting, but this was corrected by adjusting the learning rate and the number of training epochs.

### Future Improvements

- **Dataset Expansion**: In future iterations, including a more diverse range of datasets could further enhance the model’s capabilities and robustness.
- **Hyperparameter Tuning**: More extensive hyperparameter tuning could lead to better model performance, particularly for specialized tasks.

## Appendix

### Resources Used

- **Python 3.11**
- **Libraries**: Transformers, Datasets, Torch, Flask, Tkinter, Streamlit, Boto3
- **Datasets**: `conv_ai_2`, Custom Dataset
- **Applications**: VS Code, Jupyter Notebooks, NVIDIA GPU

### Key Takeaways

- Fine-tuning pre-trained models can significantly improve performance in specific applications.
- Proper data preprocessing and validation are crucial for successful model training.
- Balancing model complexity and performance is essential for creating AI solutions that are both effective and efficient.




