from peft import AutoPeftModelForCausalLM, PeftModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer
import pandas as pd
import torch
from tabulate import tabulate
import torch.nn.functional as F
import numpy as np
import sys
import wandb
import os

# Set initial configs
###############################################################################
job_id = sys.argv[1].split(".")[0]
model_no = sys.argv[2]
###############################################################################
model_name = sys.argv[3]
###############################################################################
dataset_name = sys.argv[4]
dataset_type = "population" # sample or population
mode = "private" # normal or private
sample_size = 200 if dataset_type.startswith("sample") else 2000
###############################################################################
split_token = sys.argv[5] + "\n"
aggregated = True if sys.argv[6] == "1" else False
###############################################################################
model_type = "all" if aggregated else dataset_name
output_dir = os.getcwd() +  f"/../../ft-models/{model_type}/{model_name.split('/')[1].lower()}-{dataset_type}-{mode}-4"
checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("checkpoint-")[-1]))
model_path = f"{output_dir}/{latest_checkpoint}"
###############################################################################
dataset_test_df = pd.read_csv(f"/../../data/{dataset_name}/{model_name.split('/')[1].lower()}-formatted-{dataset_type}-test.csv")
dataset_test_df = dataset_test_df.sample(n=sample_size, random_state=42)
###############################################################################

# Initialize wandb
wandb.init(
    project=f"lamp-ids-results-{dataset_type}-{mode}-4",
    name=f"run-{job_id}",
    config={
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "mode": mode,
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

# Load model
model = AutoModelForCausalLM.from_pretrained(
    os.getcwd() + f"/../../models/{model_name}",
    torch_dtype=torch.float16,
    load_in_8bit=False,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)
peft_model = PeftModel.from_pretrained(model, model_path, from_transformers=True, device_map="auto")
model = peft_model.merge_and_unload()

def generate_response(prompt):
    generation_config = GenerationConfig(
        # penalty_alpha=0.6,
        # do_sample=True,
        top_p=0.9,
        temperature=0.1,
        max_new_tokens=1,
        pad_token_id=tokenizer.pad_token_id
    )

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, generation_config=generation_config)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).split(split_token)[1]

    # Calculate NLL
    outputs = model(input_ids=generated_ids)
    logits = outputs.logits
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = generated_ids[:, 1:].contiguous()
    probs = F.softmax(shifted_logits, dim=-1)
    token_probs = probs.gather(2, shifted_labels.unsqueeze(-1)).squeeze(-1)
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shifted_labels.unsqueeze(-1)).squeeze(-1)
    prompt_len = inputs["input_ids"].shape[1]
    gen_token_probs = token_probs[:, prompt_len - 1:]
    gen_token_log_probs = token_log_probs[:, prompt_len - 1:]
    generated_tokens = generated_ids[0][prompt_len:]
    decoded_tokens = tokenizer.convert_ids_to_tokens(generated_tokens)
    total_p = gen_token_probs.sum().item()
    avg_p = gen_token_probs.mean().item()
    total_nll = -gen_token_log_probs.sum().item()
    avg_nll = -gen_token_log_probs.mean().item()

    return generated_text, total_nll, total_p

def get_optimal_threshold(scores, y_true, y_pred, num_thresholds=1000, lambda_=0.8):
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    thresholds = np.linspace(0, max(scores), num=num_thresholds)
    best_ratio = -1
    s_alpha = 0
    ratios = []

    for s_x in thresholds:
        m = np.sum((scores <= s_x) & (y_true == y_pred))
        l = np.sum(scores <= s_x)
        n = len(scores)
        if l == 0:
            continue
        precision = m / l
        coverage = l / n
        ratio = lambda_ * precision + (1 - lambda_) * coverage
        ratios.append((s_x, ratio))
        if ratio > best_ratio:
            best_ratio = ratio
            s_alpha = s_x

    return s_alpha, ratios

y_true = [str(r) for r in dataset_test_df['label'].tolist()]
generated_texts = []
y_pred = []
total_nlls = []
total_ps = []

for i in tqdm(range(len(dataset_test_df))):
    generated_text, total_nll, total_p = generate_response(dataset_test_df.iloc[i]["text"])
    generated_texts.append(generated_text)
    y_pred.append("1" if generated_text == "1" else "0")
    total_nlls.append(total_nll)
    total_ps.append(total_p)
    wandb.log({"nll": total_nll, "p": total_p})

np_y_true = np.array([int(i) for i in y_true])

np_total_nlls = np.array(total_nlls)
normalized_total_nlls = (np_total_nlls - np_total_nlls.min()) / (np_total_nlls.max() - np_total_nlls.min())

np_total_ps = np.array(total_ps)
normalized_total_ps = (np_total_ps - np_total_ps.min()) / (np_total_ps.max() - np_total_ps.min())

y_scores_nll = np.zeros((len(y_true), 2))
y_scores_p = np.zeros((len(y_true), 2))
for i, (pred, nll, p) in enumerate(zip(y_pred, normalized_total_nlls, normalized_total_ps)):
    try:
        y_scores_nll[i, int(pred)] = 1 - nll
        y_scores_nll[i, 1 - int(pred)] = nll
        y_scores_p[i, int(pred)] = p
        y_scores_p[i, 1 - int(pred)] = 1 - p
    except Exception as e:
        continue

wandb.log({"roc_nll": wandb.plot.roc_curve(y_true, y_scores_nll, labels=["Benign", "Attack"])})
wandb.log({"roc_p": wandb.plot.roc_curve(y_true, y_scores_p, labels=["Benign", "Attack"])})

# This solves both false positives and false negatives
y_pred_nll = []
s_alpha, ratios = get_optimal_threshold(total_nlls, y_true, y_pred)
for i in range(len(dataset_test_df)):
    if total_nlls[i] > s_alpha:
        y_pred_nll.append("0" if y_pred[i] == "1" else "1")
    else:
        y_pred_nll.append(y_pred[i])

results_df = pd.DataFrame({
    'y_true': y_true,
    'generated_texts': generated_texts,
    'y_pred': y_pred,
    'y_pred_nll': y_pred_nll,
    'total_nll': total_nlls,
    'total_p': total_ps
})
results_df.to_csv(os.getcwd() + f"/../../results/{dataset_name}/{model_name.split('/')[1].lower()}-{dataset_type}-{mode}-4.csv", index=False)

try:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    auc_0 = roc_auc_score(1 - np_y_true, y_scores_p[:, 0])
    auc_1 = roc_auc_score(np_y_true, y_scores_p[:, 1])
    
    acc_nll = accuracy_score(y_true, y_pred_nll)
    prec_nll = precision_score(y_true, y_pred_nll, average='macro')
    rec_nll = recall_score(y_true, y_pred_nll, average='macro')
    f1_nll = f1_score(y_true, y_pred_nll, average='macro')
    auc_nll_0 = roc_auc_score(1 - np_y_true, y_scores_nll[:, 0])
    auc_nll_1 = roc_auc_score(np_y_true, y_scores_nll[:, 1])

    table = [
        ["Without NLL", f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{auc_0:.4f}", f"{auc_1:.4f}"],
        ["With NLL", f"{acc_nll:.4f}", f"{prec_nll:.4f}", f"{rec_nll:.4f}", f"{f1_nll:.4f}", f"{auc_nll_0:.4f}", f"{auc_nll_1:.4f}"],
    ]

    wandb.run.summary["acc"] = acc
    wandb.run.summary["prec"] = prec
    wandb.run.summary["rec"] = rec
    wandb.run.summary["f1"] = f1
    wandb.run.summary["auc_0"] = auc_0
    wandb.run.summary["auc_1"] = auc_1
    
    wandb.run.summary["acc_nll"] = acc_nll
    wandb.run.summary["prec_nll"] = prec_nll
    wandb.run.summary["rec_nll"] = rec_nll
    wandb.run.summary["f1_nll"] = f1_nll
    wandb.run.summary["auc_nll_0"] = auc_nll_0
    wandb.run.summary["auc_nll_1"] = auc_nll_1

    print(tabulate(table, headers=['Method', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC-0', 'AUC-ROC-1'], tablefmt='fancy_grid', colalign=['left', 'right', 'right', 'right', 'right', 'right']))
    print()
    print(f"s_alpha: {s_alpha:.4f}")
except Exception as e:
    print(e)

wandb.finish()
