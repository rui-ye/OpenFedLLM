import argparse
import re
import json
import ast
import numpy as np

one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")


def get_socres(file_path):
    with open(file_path, "r") as f:
        ori_judge = json.load(f)

    score_list = []
    for answer_judge in ori_judge:
        judgement = answer_judge["response"]
        match = re.search(one_score_pattern, judgement)
        if not match:
            match = re.search(one_score_pattern_backup, judgement)
        if match:
            score = ast.literal_eval(match.groups()[0])
        else:
            score = -1
        score_list.append(score)
    
    print("="*100)
    print(f"File path: {file_path}")
    print(f"Total number: {len(score_list)}")
    print(f"Averaged score: {np.mean(score_list)}, std: {np.std(score_list)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_list",
        type=str,
        nargs="+",
        default=None,
        help="A list of judge results to be calculated",
    )
    args = parser.parse_args()

    for eval_name in args.eval_list:
        file_path = f"./data/vicuna/model_judgment/{eval_name}.json"
        get_socres(file_path)
