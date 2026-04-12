import os
import json
import random
random.seed(42)
import concurrent.futures
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
import time
import argparse
from task import TASKS
from prompts import WITH_KP_PROMPT_TEMPLATE, WITHOUT_KP_PROMPT_TEMPLATE

# --------------------------------------------------------------------------- #
#                   Global constants / variables                              #
# --------------------------------------------------------------------------- #
DATA_DIR    = "./data"
MAX_TOKENS  = 32768
TEMPERATURE = 0.7
TOP_P       = 0.9

# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #
def load_samples(filepath: str):
    """Read parquet file and return a list of samples with ALL fields preserved."""
    df = pd.read_parquet(filepath)

    samples = []

    for i in range(len(df)):
        row = df.iloc[i].to_dict()

        prompt = row["prompt"][0]["content"].strip()
        answer = row["reward_model"]["ground_truth"].strip()

        row["example_id"] = i
        row["prompt"] = prompt
        row["answer"] = answer

        samples.append(row)

    print(f"Total unique samples: {len(samples)}")
    return samples



def split_seeds(seeds: list[int], num_workers: int):
    """Round-robin split of the seed list into num_workers chunks."""
    chunks = [[] for _ in range(num_workers)]
    for idx, s in enumerate(seeds):
        chunks[idx % num_workers].append(s)
    return chunks


# --------------------------------------------------------------------------- #
#                           Worker process (one GPU)                          #
# --------------------------------------------------------------------------- #
def worker_process(args_tuple):
    """
    Each worker runs on a single GPU:

    args_tuple = (samples, seed_list, gpu_id)
    """
    samples, seed_list, gpu_id, model_path = args_tuple

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sleep_time = gpu_id * 8   
    print(f"[GPU {gpu_id}] sleeping {sleep_time}s before LLM init", flush=True)
    time.sleep(sleep_time)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(18000 + gpu_id)

    print(f"[GPU {gpu_id}] using MASTER_PORT={18000 + gpu_id}", flush=True)

    llm = LLM(model=model_path, enforce_eager=True)
    results = []

    for seed in seed_list:
        sampling = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            seed=seed,
        )
        messages = [[{"role": "user", "content": s["prompt"]}] for s in samples]
        outputs = llm.chat(messages, sampling, use_tqdm=True)
        for sample, out in zip(samples, outputs):
            results.append(
                {
                    "example_id": sample["example_id"],
                    "prompt": sample["prompt"],
                    "answer": sample["answer"],
                    "seed": seed,
                    "response": out.outputs[0].text,
                }
            )
    return results


# --------------------------------------------------------------------------- #
#                                   main                                      #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        help=f"Task names, choose from {list(TASKS.keys())}",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or HF name of the model",
    )
    args = parser.parse_args()

    model_path = args.model
    raw_tasks = args.tasks   # e.g. ["AIME24,AIME25,..."]

    task_keys = []
    for t in raw_tasks:
        task_keys.extend(
            [x for x in t.split(",") if x.strip()]
        )

    print(f"[INFO] Parsed task list: {task_keys}")

    OUT_DIR = Path(f"eval_outputs/{Path(model_path).name}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    available_workers = [0,1,2,3,4,5,6,7]
    num_workers = len(available_workers)

    for task_key in task_keys:
        if task_key not in TASKS:
            raise ValueError(f"Unknown task {task_key}, available: {list(TASKS.keys())}")

        task = TASKS[task_key]
        task_name = task["name"]
        task_path = task["path"]
        N = task["N"]

        print(f"\n========== Starting evaluation for task: {task_name} (N={N}) ==========")

        # Update output path for the current task
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUT_DIR / (
            f"{task_name.lower()}_t{TEMPERATURE}_p{TOP_P}_n{N}-MNT{MAX_TOKENS}_{timestamp}.jsonl"
        )

        # 1. Load original prompts
        samples = load_samples(task_path)

        # 2. Append suffix prompt to each sample
        for s in samples:
            if "CSS" in task_name:
                kp_list = s["css_selected_kps"]
            elif "CBRS" in task_name:
                kp_list = s["cbrs_selected_kps"]
            elif task_name in ['AIME24', 'AIME25', 'BRUMO25', 'HMMT25', 'AMC23', 'CMIMC25', 'MATH_500', 'Olympiad_Bench']:
                kp_list = []
            else:
                raise ValueError(f"Unknown task {task_name}")

            if len(kp_list) > 0:
                kp_str = "".join(
                    f"{i+1}. {kp.strip()}\n"
                    for i, kp in enumerate(kp_list)
                ).strip()
                s["prompt"] = WITH_KP_PROMPT_TEMPLATE.format(
                    problem=s["prompt"],
                    kp_list=kp_str,
                )
            else:
                s["prompt"] = WITHOUT_KP_PROMPT_TEMPLATE.format(
                    problem=s["prompt"]
                )

        # demo print
        print("Example prompt after formatting:")
        print(samples[0]["prompt"][:500], "...\n")

        # 3. Generate N distinct random seeds and split across GPUs
        random_seeds = random.sample(range(2**31 - 1), N)
        seed_chunks = split_seeds(random_seeds, num_workers)

        # 4. Launch workers
        all_results = []
        args_list = [
            (samples, seed_chunks[i], gid, model_path)
            for i, gid in enumerate(available_workers)
        ]

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(worker_process, tup) for tup in args_list]
            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"GPU workers ({task_name})",
            ):
                all_results.extend(fut.result())

        print(
            f"Total generations collected for {task_name}: {len(all_results)}"
        )  # len(samples) * N

        # 5. Save to disk
        with out_path.open("w", encoding="utf-8") as f:
            for item in all_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Saved results for {task_name} to {out_path}")


if __name__ == "__main__":
    main()

