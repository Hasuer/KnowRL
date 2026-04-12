import json
import os
from pathlib import Path
from collections import defaultdict
import argparse
from multiprocessing import Pool, cpu_count
from prompts import CV_PROMPT, CV_COT_PROMPT
import re
import random
from typing import List

import sys
sys.path.append("/root/paddlejob/workspace/env_run/output/yulinhao/projects/verl")
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
from verl.utils.reward_score import math_verify

from openai import OpenAI

# ============================================================
# Tasks
# ============================================================
task_name_list = [
    "AIME24","AIME25","BRUMO25","HMMT25",
    "AMC23","CMIMC25","MATH_500","Olympiad_Bench",
]

TASK_SET = {t.lower() for t in task_name_list}


# ============================================================
# Load Balancer
# ============================================================
class APILoadBalancer:
    
    def __init__(self, api_bases: List[str], api_key: str = "EMPTY"):
        self.clients = [
            OpenAI(api_key=api_key, base_url=base_url) 
            for base_url in api_bases
        ]
        self.api_bases = api_bases
        print(f"[INFO] Initialized {len(self.clients)} API clients")
        for i, base in enumerate(api_bases):
            print(f"  [{i}] {base}")
    
    def get_client(self):
        return random.choice(self.clients)
    
    def call_with_retry(self, model_name: str, messages: list, 
                       temperature: float = 0.0, max_tokens: int = 2048,
                       max_retries: int = 1):
        tried_indices = set()
        
        for attempt in range(max_retries):
            # 随机选择一个还没试过的 client
            available_indices = [i for i in range(len(self.clients)) if i not in tried_indices]
            
            if not available_indices:
                print(f"[ERROR] All APIs failed after {max_retries} attempts")
                return None
            
            idx = random.choice(available_indices)
            tried_indices.add(idx)
            client = self.clients[idx]
            
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[WARNING] API {self.api_bases[idx]} failed (attempt {attempt+1}/{max_retries}): {e}")
                continue
        
        return None


# ============================================================
# Utils
# ============================================================
def extract_solution(solution_str: str) -> str:
    if "\\boxed{" not in solution_str:
        solution_str = "\\boxed{" + solution_str + "}"
    return remove_boxed(last_boxed_only_string(solution_str))

def process_judgment(judgment_str: str) -> str:
    # First try to find the exact \boxed{letter} pattern
    boxed_matches = re.findall(r'boxed{([A-C])}', judgment_str)
    if boxed_matches:
        return boxed_matches[-1]
    
    # Directly return the judgment if it is A, B, or C
    if judgment_str in ["A", "B", "C"]:
        return judgment_str
    else:
        final_judgment_str = judgment_str.split("Final Judgment:")[-1]
        matches = re.findall(r'\(([A-C])\)*', final_judgment_str)
        if matches:
            return matches[-1]
        matches = re.findall(r'([A-C])', final_judgment_str)
        if matches:
            return matches[-1]
        return ""

def evaluate_file(input_file: str, use_model_verification: bool, 
                 load_balancer: APILoadBalancer, model_name: str):
    correct_count = 0
    total_count = 0
    
    rule_base_incorrect_samples = []

    with open(input_file, "r", encoding="utf-8") as infile:
        for idx, line in enumerate(infile):
            try:
                data = json.loads(line)

                llm_output = data["response"]

                answer = extract_solution(data["answer"])
                is_correct = bool(
                    math_verify.compute_score(llm_output, answer)
                )
                total_count += 1
                if is_correct:
                    correct_count += 1
                else:
                    if use_model_verification:
                        rule_base_incorrect_samples.append({
                            "question": data["prompt"],
                            "gold_answer": answer,
                            "llm_response": llm_output.split("</think>")[0].strip()[-10000:],
                            "idx": idx
                        })
            except Exception as e:
                print(e)
                continue

    if use_model_verification and rule_base_incorrect_samples and load_balancer is not None:
        for sample in rule_base_incorrect_samples:
            model_input = CV_PROMPT.format(
                question=sample["question"],
                gold_answer=sample["gold_answer"].split("think"),
                llm_response=sample["llm_response"]
            )
            messages = [{"role": "user", "content": model_input}]
            
            judgement = load_balancer.call_with_retry(model_name, messages)
            
            if judgement:
                judgement = process_judgment(judgement.strip())
                print(judgement)
                if judgement == 'A':
                    correct_count += 1

    if total_count == 0:
        return 0.0

    return correct_count / total_count * 100


# ============================================================
# Worker
# ============================================================
def worker(args):
    input_file, use_model_verification, api_bases, api_key, model_name = args

    filename = input_file.name
    task_name = filename.split("_t0.7")[0]

    if task_name not in TASK_SET:
        return None

    load_balancer = None
    if use_model_verification:
        try:
            load_balancer = APILoadBalancer(api_bases, api_key)
        except Exception as e:
            print(f"[WARNING] Failed to initialize load balancer: {e}")
            use_model_verification = False

    try:
        acc = evaluate_file(
            str(input_file), 
            use_model_verification, 
            load_balancer, 
            model_name
        )
        return task_name, f"{acc:.2f}"
    except Exception as e:
        print(f"[ERROR] Processing {task_name}: {e}")
        return task_name, "ERR"


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--use_model_verification",
        action="store_true",
        help="Use model verification for rule-base incorrect samples"
    )
    parser.add_argument(
        "--api_bases",
        type=str,
        nargs='+',
        default=["http://localhost:8000/v1"],
        help="OpenAI API compatible base URLs (space-separated)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API key (use 'EMPTY' for vLLM)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="cv_3b",
        help="Model name for API calls"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(cpu_count(), 16),
    )
    args = parser.parse_args()

    all_files = sorted(
        f for f in Path(args.root_dir).rglob("*.jsonl") if f.is_file()
    )

    print(f"[INFO] Found {len(all_files)} jsonl files")
    print(f"[INFO] Using {args.num_workers} workers")
    if args.use_model_verification:
        print(f"[INFO] Model verification enabled")
        print(f"[INFO] API Bases ({len(args.api_bases)} endpoints):")
        for i, base in enumerate(args.api_bases):
            print(f"  [{i}] {base}")
        print(f"[INFO] Model: {args.model_name}")

    task2acc = defaultdict(list)

    # 使用多进程处理
    with Pool(processes=args.num_workers) as pool:
        worker_args = [
            (f, args.use_model_verification, 
             args.api_bases, args.api_key, args.model_name) 
            for f in all_files
        ]
        
        for result in pool.imap_unordered(worker, worker_args, chunksize=1):
            if result is None:
                continue
            task_name, acc_str = result
            task2acc[task_name].append(acc_str)

    # ========= print results =========
    for task_name in sorted(task2acc.keys()):
        acc_str = "/".join(task2acc[task_name])
        print(f"{task_name} {acc_str}")


if __name__ == "__main__":
    main()