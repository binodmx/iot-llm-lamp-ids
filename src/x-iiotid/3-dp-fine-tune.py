from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer, SFTConfig
import pandas as pd
import torch
import sys
import os
import wandb
import transformers
import dp_transformers

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
target_epsilon = 4
output_dir = os.getcwd() + f"/../../ft-models/{dataset_name}/{model_name.split('/')[1].lower()}-{dataset_type}-private-4"
###############################################################################

wandb.init(
    project=f"lamp-ids-fine-tune-{dataset_type}-private-4",
    name=f"run-dpft-{job_id}",
    config={
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "output_dir": output_dir,
        "target_epsilon": target_epsilon
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
    bnb_4bit_use_double_quant=True
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
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['q_proj', 'v_proj']
)
model = get_peft_model(model, peft_config)

# Set training arguments
training_args = dp_transformers.TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=8,
    # max_grad_norm=1,
    # max_steps=250,
    learning_rate=2e-3,
    logging_steps=10,
    # save_steps=10,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    report_to="wandb",
    # eval_steps=4,
    # per_device_eval_batch_size=8,
    # eval_accumulation_steps=1,
    # weight_decay=0.01,
    remove_unused_columns=False,
    # dataloader_num_workers=2,
    # label_names="labels",
    # log_level="info",
    # fp16=False,
    # bf16=False,
    save_safetensors=False
)

# Set privacy arguments
privacy_args = dp_transformers.PrivacyArguments(
    per_sample_max_grad_norm=1,
    # noise_multiplier=0.1,
    target_epsilon=target_epsilon,
    # target_delta=1e-5,
    disable_dp=False
)

# Tokenize the dataset
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        # return_tensors="pt"
    )
tokenized_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)

# Initialize Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = dp_transformers.dp_utils.OpacusDPTrainer(
    args=training_args,
    model=model,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    # compute_metrics=dataset.compute_metrics,
    # preprocess_logits_for_metrics=dataset.preprocess_logits_for_metrics,
    privacy_args=privacy_args,
    data_collator=data_collator,
)

trainer.args.gradient_checkpointing = False
trainer.train()

wandb.finish()

# Save model
checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("checkpoint-")[-1]))
model.save_pretrained(f"{output_dir}/{latest_checkpoint}")
print(f"Fine-tuned model saved to {output_dir}/{latest_checkpoint}")
