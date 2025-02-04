import os
import numpy as np

epoch_range = list(range(10,11))  # 1부터 10까지 에폭 순서 지정
learning_rate = ['1e-05']
methods = ['GA+GD', 'GA+KL', 'NPO+GD', 'NPO+KL', 'MK', 'DPO+GD', 'DPO+KL', 'IDK+GD', 'IDK+KL', 'NM_JWJ']
# methods = ['GA+GD', 'GA+KL', 'DPO+GD', 'DPO+KL', 'NPO+GD', 'IDK+GD', 'IDK+KL', 'NM_JWJ', 'MK', 'IDK+AP']


def to_csv(data, filename):
    with open(filename, 'a') as f:
        f.write(','.join(data) + '\n')

def parse_single_results(lines):
    for line in lines:
        if 'Model Utility' in line:
            model_utility = line.split(': ')[1].strip()
        elif 'Forget Efficacy' in line:
            forget_efficacy = line.split(': ')[1].strip()
        elif 'Retain ROUGE' in line:
            retain_rouge = line.split(': ')[1].strip()
        elif 'Real Authors ROUGE' in line:
            authors_rouge = line.split(': ')[1].strip()
        elif 'Real World ROUGE' in line:
            world_rouge = line.split(': ')[1].strip()
    return [model_utility, forget_efficacy]

def parse_mixed_results(lines):
    for line in lines:
        if '[Single] Retain Score' in line:
            retain_score = line.split(': ')[1].strip()
        elif '[Single] Forget Score' in line:
            forget_score = line.split(': ')[1].strip()
        elif '[Retain-Retain] 1st Retain Score' in line:
            retain_retain_1 = line.split(': ')[1].strip()
        elif '[Retain-Retain] 2nd Retain Score' in line:
            retain_retain_2 = line.split(': ')[1].strip()
        elif '[Forget-Forget] 1st Forget Score' in line:
            forget_forget_1 = line.split(': ')[1].strip()
        elif '[Forget-Forget] 2nd Forget Score' in line:
            forget_forget_2 = line.split(': ')[1].strip()
        elif '[Retain-Forget] Retain Score' in line:
            retain_forget_retain = line.split(': ')[1].strip()
        elif '[Retain-Forget] Forget Score' in line:
            retain_forget_forget = line.split(': ')[1].strip()
        elif '[Forget-Retain] Retain Score' in line:
            forget_retain_retain = line.split(': ')[1].strip()
        elif '[Forget-Retain] Forget Score' in line:
            forget_retain_forget = line.split(': ')[1].strip()
        elif 'Mean Retain Score' in line:
            mean_retain = line.split(': ')[1].strip()
        elif 'Mean Forget Score' in line:
            mean_forget = line.split(': ')[1].strip()
    return [retain_score, forget_score, retain_retain_1, retain_retain_2, forget_forget_1, forget_forget_2, retain_forget_retain, retain_forget_forget, forget_retain_retain, forget_retain_forget], [(float(forget_retain_retain) + float(retain_forget_retain) - float(retain_forget_forget) - float(forget_retain_forget)) / 10]

top = "results_WT_TEST7/tofu/llama2-7b/"
end_dir = "results.csv"

# 초기화 (첫 줄에 헤더 추가)
with open(end_dir, 'w') as f:
    # f.write(','.join(['Method', 'Epoch', 'MU', 'FE', 'RS', 'FS', 'R-R1', 'R-R2', 'F-F1', 'F-F2', 'R-F(R)', 'R-F(F)', 'F-R(R)', 'F-R(F)', 'Mean R', 'Mean F']) + '\n')
    # f.write(','.join(['MU', 'FE', 'Method', 'Epoch', 'ROUGE(only_retain)', 'ROUGE(authors)', 'ROUGE(world)']) + '\n')
    f.write(','.join(['Method', 'Epoch', 'MU', 'FE']) + '\n')

# 중복 제거를 위한 집합 & 정렬을 위한 딕셔너리
visited = set()
results_dict = {method: {} for method in methods}  # { method: {epoch: data} }

for root, dirs, files in os.walk(top):
    for file in files:
        flag = False
        for epoch in epoch_range:
            for lr in learning_rate:
                if f'epoch{epoch}_{lr}' in root:
                    flag = True
                    current_epoch = epoch
                    current_lr = lr
        
        if not flag:
            continue

        flag = False
        if 'forget05' not in root:
            continue

        found_method = None
        for method in methods:
            if f'/{method}/' in root:
                flag = True
                current_method = method

        if not flag:
            continue

        # 중복 확인 (Method, Epoch 기준으로)
        key = (current_method, current_epoch)
        if key in visited:
            continue  # 중복이면 스킵
        visited.add(key)  # 새로운 데이터만 추가

        if 'unlearn_times_1' in dirs:
            current_analysis = f'{root}/unlearn_times_1/eval_results-last'
            unlearning_txt_dir = f'{current_analysis}/unlearning_results.txt'
            mixed_txt_dir = f'{current_analysis}/mixed_results_gpt.txt'

            with open(unlearning_txt_dir, 'r') as f:
                unlearning_lines = f.readlines()
                single_results = parse_single_results(unlearning_lines)

            try:
                with open(mixed_txt_dir, 'r') as f:
                    mixed_lines = f.readlines()
                    mixed_results, concise_results = parse_mixed_results(mixed_lines)
            except:
                mixed_results = ['-10'] * 11
                concise_results = ['-10']

            single_results = [str(round(float(x), 4)) for x in single_results]
            mixed_results = [str(round(float(x), 4)) for x in mixed_results]
            concise_results = [str(round(float(x), 4)) for x in concise_results]

            # average = str(round(np.mean([float(x) for x in single_results + mixed_results]).item(), 4))

            # 정렬을 위해 results_dict에 저장
            results_dict[current_method][current_epoch] = single_results + mixed_results 

# CSV 저장 (에폭 순서대로)
for method in methods:
    for epoch in sorted(results_dict[method].keys()):  # 에폭 정렬
        to_csv([method, str(epoch)] + results_dict[method][epoch], end_dir)
