# LAMP-IDS: LLM-assisted Multi-modal and Privacy-aware Collaborative Perception for IoT Network Intrusion Detection

## Introduction

In this research, we propose LAMP-IDS, a novel framework that leverages Large Language Models (LLMs) for privacy-aware collaborative perception in Network Intrusion Detection Systems (NIDS). The core of our contribution is a novel fine-tuning methodology that integrates Differential Privacy (DP) with Low-Rank Adaptation (LoRA). This technique allows for the secure aggregation and fusion of diverse network data from multiple sources, enabling collaborative model training without exposing sensitive information. Furthermore, we introduce an optimized prediction algorithm that uses Negative Log-Likelihood (NLL) scores to classify network traffic, which enhances detection accuracy by leveraging the model's confidence to identify anomalous patterns. We conducted a comprehensive empirical evaluation using three foundational LLMs and three distinct IoT network datasets. The results demonstrate that LAMP-IDS significantly improves intrusion detection performance through collaborative perception compared to models trained on individual datasets, all while adhering to formal privacy guarantees.

## Getting Started

1. Setup a python v3.10+ environment and set environment variables.
    ```
    WANDB_API_KEY=
    WANDB_SILENT=true
    HF_TOKEN=
    ```
2. Install python packages listed in `requirements.txt`.
3. Install modified `dp-transformers` module using following command.
    ```bash
    pip install git+https://github.com/binodmx/dp-transformers.git
    ```
4. Download the relevant datasets and place them under `/data` directory.
    ```
    data
      ├──cic-iot-2023
      ├──wustl-iiot
      └──x-iiotid
    ```
5. Download the relevant large language models and place them under `/models` directory.
    ```
    models
      ├──google
      │    └──gemma-3-1b-it
      ├──meta-llama
      │    └──Llama-3.2-1B-Instruct
      └──TinyLlama
           └──TinyLlama-1.1B-Chat-v1.0
    ```
6. Pre-process the datasets using `<dataset_name>/1-create-dataset.ipynb` notebooks.
7. Execute the normal and dp fine-tune.
    ```bash
    python3 main.py $PBS_JOBID "google/gemma-3-1b-it" "x-iiotid" > ./logs/$PBS_JOBID.log
    ```
8. Observe the results saved into wandb.
