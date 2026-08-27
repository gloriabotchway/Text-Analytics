import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

CLEAN_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned_dataset.csv")
FINETUNE_DIR = os.path.join(PROJECT_ROOT, "data", "mlx_finetune")

SYSTEM_PROMPT = "You are a helpful, concise customer-service assistant for an e-commerce company. Answer only using information the customer provides in their message (order numbers, account details, etc). Never invent order numbers, dates, or account information."

BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
ADAPTER_DIR = "llama_customer_service_lora_adapters"


def build_chat_example(instruction, response):
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]}


def build_finetune_dataset():
    df = pd.read_csv(CLEAN_PATH)
    os.makedirs(FINETUNE_DIR, exist_ok=True)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["intent"])
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["intent"])

    splits = {"train": train_df, "valid": valid_df, "test": test_df}
    counts = {}
    for split_name in splits:
        split_df = splits[split_name]
        path = os.path.join(FINETUNE_DIR, split_name + ".jsonl")
        f = open(path, "w")
        for row in split_df.itertuples():
            example = build_chat_example(row.instruction, row.response)
            f.write(json.dumps(example) + "\n")
        f.close()
        counts[split_name] = len(split_df)

    print("MLX-LM fine-tuning data written to " + FINETUNE_DIR + "/")
    for split_name in counts:
        print("  " + split_name + ".jsonl : " + str(counts[split_name]) + " examples")
    return counts


if __name__ == "__main__":
    build_finetune_dataset()
    print("")
    print("Data is ready. Next, run this in your terminal:")
    print("")
    print("mlx_lm.lora --model " + BASE_MODEL + " --train --data " + FINETUNE_DIR + " --adapter-path " + ADAPTER_DIR + " --iters 50 --batch-size 4 --learning-rate 1e-5 --num-layers 8")