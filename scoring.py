import jsonlines
from collections import defaultdict
import re
import pprint

forget = 'forget01'
# lr = "0.0005"
# method = "IDK+AP"
# method = "NPO+GD"
# method = "Retrain"

lr = "0.0002"
# method = "DPO+GD"
method = "ME+GD"

results = defaultdict(list)

with jsonlines.open(f"generated_data/{forget}/{method}_{lr}_score.jsonl") as reader:
    lines = list(reader)


for idx, line in enumerate(lines):
    for key in line.keys():
        if 'score' in key:
            try: 
                try: 
                    results[key].append(int(line[key]))
                except:
                    numbers = re.findall(r'\d+', line[key])
                    int_list = [int(num) for num in numbers]
                    results[f"{key}_forget"].append(int_list[0])
                    results[f"{key}_retain"].append(int_list[1])
            except:
                print("ERROR " + str(idx) + ", " + str(results[key]))

del results['mixed_score_top1']
del results['mixed_score_top5']
del results['mixed_score_top1800']
del results['mixed_score_top3600']

del results['mixed_score_reversed_top1']
del results['mixed_score_reversed_top5']
del results['mixed_score_reversed_top1800']
del results['mixed_score_reversed_top3600']

del results['retain_retain_score']

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black']
print(results.keys())

# 평균
for key in results:
    if 'forget' in key or 'retain' in key:
        results[key] = round(sum(results[key]) / len(results[key]), 3)

mixed_retain = [results[f'mixed_score_top{k}_retain'] for k in [1,5,1800,3600]]
mixed_forget = [results[f'mixed_score_top{k}_forget'] for k in [1,5,1800,3600]]
mixed_reversed_retain = [results[f'mixed_score_reversed_top{k}_retain'] for k in [1,5,1800,3600]]
mixed_reversed_forget = [results[f'mixed_score_reversed_top{k}_forget'] for k in [1,5,1800,3600]]
retain_retain = [results['retain_retain_score_forget'], results['retain_retain_score_retain']]
retain = [results[f"retain_score_top{k}"] for k in [1,5,1800,3600]]

forget_score = results['forget_score']

print("Mixed Retain: ", mixed_retain)
print("Mixed Forget: ", mixed_forget)
print("Mixed Reversed Retain: ", mixed_reversed_retain)
print("Mixed Reversed Forget: ", mixed_reversed_forget)
print("Retain Retain: ", retain_retain)
print("Retain: ", retain)
print("Forget Score: ", forget_score)

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee'])
x = ['Sim 0001', 'Sim 0005', 'Sim 1800', 'Sim 3600']
plt.figure(figsize=(3, 2))
plt.plot(x, retain, label="Retain", marker="*", color='red', linestyle='-')
# plt.plot(x, mixed_retain, label="(Forget + Retain) Retain", marker="o", color="#dd0000", linestyle='--')
# plt.plot(x, mixed_forget, label="(Forget + Retain) Forget", marker="o", color="#555555", linestyle='--')
# plt.plot(x, mixed_reversed_retain, label="(Retain + Forget) Retain", marker="s", color="#dd0000", linestyle='-.')
# plt.plot(x, mixed_reversed_forget, label="(Retain + Forget) Forget", marker="s", color="#555555", linestyle='-.')
# mean reversed and mixed
plt.plot(x, [(summed[0] + summed[1])/2 for summed in zip(mixed_retain, mixed_reversed_retain)], label="Mean Retain", marker="*", color='blue', linestyle='-')
plt.plot(x, [(summed[0] + summed[1])/2 for summed in zip(mixed_forget, mixed_reversed_forget)], label="Mean Forget", marker="*", color='green', linestyle='-')

plt.plot(x, [forget_score, forget_score, forget_score, forget_score], label="Forget", marker="*", color='black', linestyle='-')
# plt.plot(x, [retain_retain[0], retain_retain[0], retain_retain[0], retain_retain[0]], label="Retain Retain Front", marker="*", color='#ff7f0e')
# plt.plot(x, [retain_retain[1], retain_retain[1], retain_retain[1], retain_retain[1]], label="Retain Retain Back", marker="*", color='pink')

plt.gca().yaxis.set_ticks_position('left')  # y축 tick을 왼쪽에만 표시
plt.gca().xaxis.set_ticks_position('bottom')  # x축 t
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tick_params(axis='both', which='minor', length=0)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel("Retain Sample",)
plt.ylabel("LLM-as-a-judge Score")
plt.ylim(0, 10)
plt.savefig(f"{method}_{forget}_clean.png", dpi=2000)