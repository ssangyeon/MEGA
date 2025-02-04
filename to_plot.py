import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scienceplots

plt.style.use('science')

learning_rate = ['1e-05']
methods = ['GA+GD', 'GA+KL', 'NPO+GD', 'NPO+KL', 'ME+GD', 'MK', 'DPO+GD', 'DPO+KL', 'IDK+GD', 'IDK+KL', 'IDK+AP', 'NM_JWJ']
# methods = ['NM_JWJ', 'MK', 'GA+GD', 'IDK+GD']
def parse_single_results(lines):
    model_utility = None
    for line in lines:
        if 'Model Utility_base' in line:
            model_utility = line.split(': ')[1].strip()
    return float(model_utility) if model_utility else None

def parse_mixed_results(lines):

    forget_retain_retain = None
    retain_forget_retain = None
    retain_forget_forget = None
    forget_retain_forget = None

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

    if forget_score and forget_retain_retain and retain_forget_retain and retain_forget_forget and forget_retain_forget:
        single_retain = float(retain_score) / 10
        single_forget = 1 - float(forget_score) / 10
        retain_val = (float(forget_retain_retain) + float(retain_forget_retain)) / 20
        forget_val = (float(retain_forget_forget) + float(forget_retain_forget)) / 20
        return single_retain, single_forget, retain_val, forget_val
    else:
        return None, None, None, None

top = "results_WT_TEST7/tofu/llama2-7b/"

# (epoch, lr, method) -> (single, retain, forget)
results_dict = {}

# 중복 처리를 위해 visited set 사용
visited = set()

for root, dirs, files in os.walk(top):
    # 일단 epoch, lr 찾기
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
    if not found_epoch_lr:
        continue
    
    # forget01 조건
    if 'forget01' not in root:
        continue
    
    # method 찾기
    found_method = False
    current_method = None
    for method in methods:
        if f'/{method}/' in root:
            found_method = True
            current_method = method
            break
    if not found_method:
        continue
    
    # 이미 방문했는지 체크 (epoch, lr, method 조합으로 중복 제거)
    key = (current_epoch, current_lr, current_method)
    if key in visited:
        continue

    # unlearn_times_1 폴더가 있는지
    if 'unlearn_times_1' in dirs:
        visited.add(key)  # 방문 처리

        current_analysis = f'{root}/unlearn_times_1/eval_results-last'
        unlearning_txt_dir = os.path.join(current_analysis, 'unlearning_results.txt')
        mixed_txt_dir      = os.path.join(current_analysis, 'mixed_results_gpt.txt')

        # 1) unlearning_results.txt (Single Score)
        single_val = None
        # if os.path.isfile(unlearning_txt_dir):
        #     with open(unlearning_txt_dir, 'r') as f:
        #         unlearning_lines = f.readlines()
        #         single_val = parse_single_results(unlearning_lines) #유틸리티

        # 2) mixed_results.txt (Retain/Forget Score)
        retain_val = None
        forget_val = None
        if os.path.isfile(mixed_txt_dir):
            with open(mixed_txt_dir, 'r') as f:
                mixed_lines = f.readlines()
                single_val, forget_efficacy, retain_val, forget_val = parse_mixed_results(mixed_lines) ##Retain score, Forget score

        if forget_efficacy is None:
            forget_efficacy = -1
        if single_val is None:
            single_val = -1
        if retain_val is None:
            retain_val = -1
        if forget_val is None:
            forget_val = -1

        results_dict[key] = (forget_efficacy, single_val, retain_val, forget_val)


grouped_data = defaultdict(dict)  
# grouped_data[(lr, method)][epoch] = (single_val, retain_val, forget_val)

for (epoch, lr, method), (efficacy, single, retain, forget) in results_dict.items():
    grouped_data[(lr, method)][epoch] = (efficacy, single, retain, forget)




# Legend 저장을 위한 빈 Figure 생성
fig_legend = plt.figure(figsize=(6, 1))
ax = fig_legend.add_subplot(111)
ax.plot([], [], marker='o', label='Forget Efficacy')
ax.plot([], [], marker='o', label='Model Utility')
ax.plot([], [], marker='o', label='Retain Score')
ax.plot([], [], marker='o', label='Forget Score')
ax.legend(loc='center', fontsize=10, ncol=4)
ax.axis('off')

legend_output_path = "/mnt/hdd0/home/aiisl/MEGA/legend.pdf"
fig_legend.savefig(legend_output_path, bbox_inches='tight')
plt.close(fig_legend)  # Figure 닫기



# 이제 (lr, method) 조합별로 플롯 그리기
for (lr, method), epoch_dict in grouped_data.items():
    # epoch_dict: { epoch: (single, retain, forget), ... }
    epochs = sorted(epoch_dict.keys())
    
    efficacy_vals = [epoch_dict[e][0] for e in epochs]
    single_vals = [epoch_dict[e][1] for e in epochs]
    retain_vals = [epoch_dict[e][2] for e in epochs]
    forget_vals = [epoch_dict[e][3] for e in epochs]

    plt.figure(figsize=(6,4))
    plt.plot(epochs, efficacy_vals, marker='o', label='Forget Efficacy')
    plt.plot(epochs, single_vals, marker='o', label='Model Utility')
    plt.plot(epochs, retain_vals, marker='o', label='Retain Score')
    plt.plot(epochs, forget_vals, marker='o', label='Forget Score')
    
    # plt.title(f"Scores vs. Epoch (LR={lr}, Method={method})")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path = f"/mnt/hdd0/home/aiisl/MEGA/score_{method}.pdf"
    plt.savefig(output_path)
    plt.show()
