import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# 1) 기본 설정
# -----------------------------
learning_rate = ['1e-05']  # LR 후보들
methods = [
    'GA+GD', 'GA+KL', 'NPO+GD', 'NPO+KL', 
    'MK', 'DPO+GD', 'DPO+KL', 
    'IDK+GD', 'IDK+KL', 'NM_JWJ'
]

# 결과 파일들이 들어있는 루트 폴더
top = "results_WT_TEST7/tofu/llama2-7b/"

# -----------------------------
# 2) 파싱 함수 정의
# -----------------------------
def parse_single_results(lines):
    """
    unlearning_results.txt 파일 내에서
    'Model Utility' 값만 추출해 float으로 반환
    """
    model_utility = None
    for line in lines:
        if 'Model Utility' in line:
            model_utility = line.split(': ')[1].strip()
    return float(model_utility) if model_utility else None

def parse_mixed_results(lines):
    """
    mixed_results_gpt.txt 파일 내에서
    retain/forget 관련 점수들을 추출해 반환
    (각 텍스트 라인에 특정 키워드가 들어있음)
    """
    # 일단 None으로 초기화
    retain_score          = None
    forget_score          = None
    retain_retain_1       = None
    retain_retain_2       = None
    forget_forget_1       = None
    forget_forget_2       = None
    retain_forget_retain  = None
    retain_forget_forget  = None
    forget_retain_retain  = None
    forget_retain_forget  = None

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

    # 모든 데이터가 존재해야 유효한 계산 가능
    if (retain_retain_1 is not None and
        retain_forget_retain is not None and
        retain_retain_2 is not None and
        forget_retain_retain is not None and
        forget_forget_1 is not None and
        forget_retain_forget is not None and
        forget_forget_2 is not None and
        retain_forget_forget is not None):
        
        # 문제에서 주어진대로 계산
        # 20은 대략 ngram 계산 시 normalizing한 것으로 추정
        retain_first  = (float(retain_retain_1) + float(retain_forget_retain)) / 2
        retain_last   = (float(retain_retain_2) + float(forget_retain_retain)) / 2
        forget_first  = (float(forget_forget_1) + float(forget_retain_forget)) / 2
        forget_last   = (float(forget_forget_2) + float(retain_forget_forget)) / 2
        single        = (float(retain_score)    - float(forget_score)) 
        return retain_last - forget_last, retain_first - forget_first, single
    else:
        return None, None, None

# -----------------------------
# 3) 폴더 트리 탐색하면서 결과 파싱
# -----------------------------
results_dict = {}  # (epoch, lr, method) -> (retain_first, retain_last, forget_first, forget_last)
visited = set()    # 중복처리용

for root, dirs, files in os.walk(top):
    # (1) epoch, lr 찾기
    found_epoch_lr = False
    current_epoch, current_lr = None, None

    for epoch in range(1, 11):
        for lr in learning_rate:
            if f'epoch{epoch}_{lr}' in root:
                found_epoch_lr = True
                current_epoch = epoch
                current_lr = lr
                break
        if found_epoch_lr:
            break
    
    # 못 찾으면 스킵
    if not found_epoch_lr:
        continue
    
    # (2) forget01 조건
    if 'forget01' not in root:
        continue
    
    # (3) method 찾기
    found_method = False
    current_method = None
    for method in methods:
        if f'/{method}/' in root:
            found_method = True
            current_method = method
            break
    if not found_method:
        continue

    # (4) 이미 (epoch, lr, method) 조합으로 방문했는지 체크
    key = (current_epoch, current_lr, current_method)
    if key in visited:
        continue
    
    # (5) unlearn_times_1 폴더가 있다면 결과 파싱
    if 'unlearn_times_1' in dirs:
        visited.add(key)

        # 해당 폴더로 접근
        current_analysis = os.path.join(root, 'unlearn_times_1', 'eval_results-last')
        unlearning_txt_dir = os.path.join(current_analysis, 'unlearning_results.txt')
        mixed_txt_dir      = os.path.join(current_analysis, 'mixed_results_gpt.txt')

        # unlearning_results.txt 에서 single_val 추출 (필요시 사용)
        single_val = None
        if os.path.isfile(unlearning_txt_dir):
            with open(unlearning_txt_dir, 'r') as f:
                unlearning_lines = f.readlines()
                single_val = parse_single_results(unlearning_lines)

        # mixed_results_gpt.txt 에서 retain/forget 점수 추출
        rfirst, rlast, ffirst, flast = -1, -1, -1, -1  # None일 경우 -1로 대체
        if os.path.isfile(mixed_txt_dir):
            with open(mixed_txt_dir, 'r') as f:
                mixed_lines = f.readlines()
                parsed = parse_mixed_results(mixed_lines)
                if parsed is not None:
                    p_rfirst, p_rlast, p_ffirst = parsed
                    if p_rfirst is not None: 
                        rfirst = p_rfirst
                    if p_rlast is not None: 
                        rlast  = p_rlast
                    if p_ffirst is not None: 
                        ffirst = p_ffirst


        # (epoch, lr, method) 기준으로 저장
        results_dict[key] = (rfirst, rlast, ffirst)

# -----------------------------
# 4) grouped_data로 구조 정리
# -----------------------------
# grouped_data[(lr, method)][epoch] = (retain_first, retain_last, forget_first, forget_last)
grouped_data = defaultdict(dict)
for (epoch, lr, method), (retain_first, retain_last, forget_first) in results_dict.items():
    grouped_data[(lr, method)][epoch] = (retain_first, retain_last, forget_first)

# -----------------------------
# 5) epoch=10, 특정 lr='1e-05'만 모아서 하나의 그래프에 표시
# -----------------------------
target_lr = '1e-05'
target_epoch = 10

# 순서 유지
methods_order = [
    'GA+GD', 'GA+KL', 'NPO+GD', 'NPO+KL', 
    'MK', 'DPO+GD', 'DPO+KL', 
    'IDK+GD', 'IDK+KL', 'NM_JWJ'
]

retain_first_vals = []
retain_last_vals  = []
forget_first_vals = []
forget_last_vals  = []

for method in methods_order:
    if (target_lr, method) in grouped_data and target_epoch in grouped_data[(target_lr, method)]:
        r_first, r_last, f_first = grouped_data[(target_lr, method)][target_epoch]
    else:
        # 해당 데이터가 없으면 0으로 처리
        r_first, r_last, f_first = (0, 0, 0)

    retain_first_vals.append(r_first)
    retain_last_vals.append(r_last)
    forget_first_vals.append(f_first)

# -----------------------------
# 6) Grouped Bar Chart
# -----------------------------
x = np.arange(len(methods_order))
bar_width = 0.2

plt.figure(figsize=(10, 6))

# 막대 4개씩 묶음으로 표현
plt.bar(x - 1.5*bar_width, retain_first_vals, width=bar_width, label='retain-forget_last')
plt.bar(x - 0.5*bar_width, retain_last_vals,  width=bar_width, label='retain-forget_first')
plt.bar(x + 0.5*bar_width, forget_first_vals, width=bar_width, label='retain-forget_single')

plt.xticks(x, methods_order, rotation=45, ha='right')
plt.ylabel("Score")
plt.title(f"Epoch={target_epoch} (LR={target_lr}) - Retain/Forget Scores by Method")
plt.legend()
plt.tight_layout()

# 결과 저장
plt.savefig("my_plot_epoch10_all_methods.pdf")
plt.show()
