import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import PeftModel
from peft import LoraConfig, TaskType, get_peft_model
from peft import prepare_model_for_kbit_training
import numpy as np
from transformers import DataCollatorWithPadding,DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_name = "unsloth/llama-3-8b-bnb-4bit"  # Example of a 4-bit quantized model
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"  # Example of a 8-bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={'': torch.cuda.current_device()},
    load_in_8bit=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("poornima9348/finance-alpaca-1k-test")

train_test_split = dataset['test'].train_test_split(test_size=0.2, shuffle=True)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# print(dataset, dataset.keys())

train_dataset = train_dataset.remove_columns(['input', 'text'])
test_dataset = test_dataset.remove_columns(['input', 'text'])

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

def prompt_builder(row):
    return {"text": row["instruction"] + row["output"]}

train_dataset =train_dataset.map(prompt_builder)
test_dataset =test_dataset.map(prompt_builder)
print("Train dataset example:", train_dataset[0])
print("Test dataset example:", test_dataset[0])
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Convert datasets to PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
train_dataset = train_dataset.remove_columns(['instruction', 'output'])
test_dataset = test_dataset.remove_columns(['instruction', 'output'])
peft_model = prepare_model_for_kbit_training(model)


lora_config = LoraConfig(
    r=32, #the rank of the update matrices, expressed in int. Lower rank results in smaller update matrices with fewer trainable parameters
    lora_alpha=32, #LoRA scaling factor
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    #target_modules='all-linear' # The modules (for example, attention blocks) to apply the LoRA update matrices.
)
lora_model = get_peft_model(peft_model, lora_config)
lora_model.print_trainable_parameters()

trainer = Trainer(
    model=lora_model,
    args=TrainingArguments(
        output_dir="Meta-Llama-3.1-8B-Instruct-finetuned",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_strategy="no",  # Disable evaluation during training to avoid NCCL issues
        save_strategy="steps",  # Save based on training steps instead of epochs
        save_steps=50,  # Save every 50 steps (adjust based on your needs)
        save_total_limit=10,  # Keep only the last 5 checkpoints to save disk space
        num_train_epochs=10,  # Number of epochs to train
        weight_decay=0.01,
        load_best_model_at_end=False,  # Set to False since we disabled eval_strategy
        logging_steps=10,  # Log every 10 steps
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        dataloader_num_workers=0
    ),
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()

# After training, you can load from specific checkpoints:

# # Load from step 100
# model_step_100 = PeftModel.from_pretrained(model, "Meta-Llama-3.1-8B-Instruct-finetuned/checkpoint-100")

# # Load from step 200
# model_step_200 = PeftModel.from_pretrained(model, "Meta-Llama-3.1-8B-Instruct-finetuned/checkpoint-200")

# # Compare performance at different steps
# def test_checkpoint_performance(base_model, checkpoint_path, tokenizer, test_prompt):
#     """Test model performance at a specific checkpoint"""
#     model_checkpoint = PeftModel.from_pretrained(base_model, checkpoint_path)
#     model_checkpoint.eval()
    
#     inputs = tokenizer(test_prompt, return_tensors="pt")
#     device = next(model_checkpoint.parameters()).device
#     inputs = {k: v.to(device) for k, v in inputs.items()}
    
#     with torch.no_grad():
#         outputs = model_checkpoint.generate(**inputs, max_new_tokens=100, temperature=0.7)
    
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(f"Checkpoint {checkpoint_path}:")
#     print(f"Response: {response}\n")
    
#     return response

# # Test different checkpoints
# test_prompt = "How does a brokerage firm work?"
# checkpoints = ["checkpoint-50", "checkpoint-100", "checkpoint-150"]

# for checkpoint in checkpoints:
#     if os.path.exists(f"Meta-Llama-3.1-8B-Instruct-finetuned/{checkpoint}"):
#         test_checkpoint_performance(model, f"Meta-Llama-3.1-8B-Instruct-finetuned/{checkpoint}", tokenizer, test_prompt)

