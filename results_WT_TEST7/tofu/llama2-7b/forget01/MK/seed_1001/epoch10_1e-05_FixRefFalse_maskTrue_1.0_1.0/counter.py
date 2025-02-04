import os
import json
from pathlib import Path

def load_jsonl(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                data.append(json.loads(line))
    else: Path(file_path).touch()
    return data

data = load_jsonl('1/unlearn_times_1/eval_results-last/evaluated.jsonl')

mean_f_score = []
mean_r_score = []
for line in data:
    r_score = int(line['retain_score'])
    f_score = int(line['forget_score'])

    mean_f_score.append(f_score)
    mean_r_score.append(r_score)

print(data)

print(1-sum(mean_f_score) / len(mean_f_score))

