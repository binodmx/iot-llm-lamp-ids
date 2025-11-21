from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, SFTConfig
import pandas as pd
import torch
import sys
import wandb
import os

# Set initial configs
###############################################################################
job_id = sys.argv[1].split(".")[0]
###############################################################################
model_name = sys.argv[2]
###############################################################################
dataset_name = sys.argv[3]
dataset_type = "population"
sample_size = 800 if dataset_type.startswith("sample") else 8000
###############################################################################
if dataset_name == "all":
    dataset_df1 = pd.read_csv(os.getcwd() + f"/../../data/x-iiotid/{model_name.split('/')[1].lower()}-formatted-{dataset_type}-train.csv")
    dataset_df2 = pd.read_csv(os.getcwd() + f"/../../data/cic-iot-2023/{model_name.split('/')[1].lower()}-formatted-{dataset_type}-train.csv")
    dataset_df3 = pd.read_csv(os.getcwd() + f"/../../data/wustl-iiot/{model_name.split('/')[1].lower()}-formatted-{dataset_type}-train.csv")
    dataset_df1 = dataset_df1.sample(n=sample_size, random_state=42)
    dataset_df2 = dataset_df2.sample(n=sample_size, random_state=42)
    dataset_df3 = dataset_df3.sample(n=sample_size, random_state=42)
    dataset_df = pd.concat([dataset_df1, dataset_df2, dataset_df3], ignore_index=True).sample(frac=1).reset_index(drop=True)
else:
    dataset_df = pd.read_csv(os.getcwd() + f"/../../data/{dataset_name}/{model_name.split('/')[1].lower()}-formatted-{dataset_type}-train.csv")
    dataset_df = dataset_df.sample(n=sample_size, random_state=42)
train_dataset = Dataset.from_pandas(dataset_df)
###############################################################################
output_dir = os.getcwd() + f"/../../ft-models/{dataset_name}/{model_name.split('/')[1].lower()}-{dataset_type}-normal"
###############################################################################

# Initialize wandb
wandb.init(
    project=f"lamp-ids-fine-tune-{dataset_type}-normal",
    name=f"run-nft-{job_id}",
    config={
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "output_dir": output_dir,
    }
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.getcwd() + f"/../../models/{model_name}",
    local_files_only=True
)

# Set pad_token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    os.getcwd() + f"/../../models/{model_name}",
    device_map="auto",
    local_files_only=True,
    quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
model.config.use_cache=False
model.config.pretraining_tp=1

# Set PEFT config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['q_proj', 'v_proj']
)

# Set training arguments
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    # gradient_accumulation_steps=8,
    num_train_epochs=1,
    # max_grad_norm=1,
    # max_steps=250,
    learning_rate=2e-4,
    logging_steps=10,
    # save_steps=50,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    report_to="wandb",
    packing=False,
    dataset_text_field="text",
    # max_seq_length=1024,
    # train_batch_size=16,
    # optim="paged_adamw_32bit",
    # fp16=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    peft_config=peft_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

trainer.train()

print(f"Fine-tuned model saved to {output_dir}")
wandb.finish()
