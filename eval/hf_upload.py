from datasets import Dataset, DatasetDict
import json
from pathlib import Path

root = Path("/root/paddlejob/workspace/env_run/output/yulinhao/projects/KnowRL/eval/data")

dataset_dict = {}

for dataset_dir in root.iterdir():
    if not dataset_dir.is_dir():
        continue

    dataset_name = dataset_dir.name
    jsonl_files = list(dataset_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"Skip {dataset_name}, no jsonl found")
        continue

    file_path = jsonl_files[0]

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    dataset = Dataset.from_list(data)

    # ⚠️ 关键：每个数据集作为一个 split
    dataset_dict[dataset_name] = dataset

hf_dataset = DatasetDict(dataset_dict)

hf_dataset.push_to_hub("HasuerYu/KnowRL-KP-Annotations")