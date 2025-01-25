import os
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
from peft import PeftModel
import argparse

def get_format_string(lists):
    Qs = ""
    for i, item in enumerate(lists):
        Qs += f"[{i+1}] {item}\n"
    inputs = f"""[INST] Prompt:
Below is a list of questions. Please answer them in order, using the format shown below. Number each answer on a new line, starting with [1] for the first question, [2] for the second, and so on.

Questions:
{Qs}

Answer format:
[1] Your answer to question 1
[2] Your answer to question 2
[3] Your answer to question 3
[...]

Please strictly follow the format above when answering the questions. 
[/INST] [ANSWER] """
    return inputs


def main(method_name, forget, lr, epoch):
    if method_name == "Original": model_name = "meta-llama/Llama-2-7b-chat-hf"
    elif method_name == "Retrain": model_name = "locuslab/tofu_ft_llama2-7b"
    else:
        if method_name == "ME+GD": reg = '0.1'; mask='False'
        else: reg = '1.0'; mask='True'
        model_name = f"results_WT/tofu/llama2-7b/{forget}/{method_name}/seed_1001/epoch{epoch}_{lr}_FixRefFalse_mask{mask}_{reg}_1.0/1/unlearn_times_1/checkpoint-last"

    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = model.eval()

    data = load_jsonl(f"data/tofu/output.jsonl")

    for data_point in data:
        inputs = []
        # for forget_count in [1,2,4]:
        #     inputs += [get_format_string(data_point["forget_question"][:forget_count])]
        # for retain_count in [1,2,4,10]:
        #     inputs += [get_format_string(data_point["retain_question"][:retain_count])]
        for forget_count in [1,2,4]:
            for retain_count in [1,2,4]:
                inputs += [get_format_string(data_point[f"forget_question"][:forget_count] + data_point["retain_question"][:retain_count])]

        # for forget_count in [1,2,4]:
        #     for wellknown_count in [1,2,4,10]:
        #         inputs += [get_format_string(data_point[f"forget_question"][:forget_count] + data_point["wellknown_question"][:wellknown_count])]

        for retain_count in [1,2,4]:
            for forget_count in [1,2,4]:
                inputs +=  [get_format_string(data_point['retain_question'][:retain_count] + data_point['forget_question'][:forget_count])]
        
        # for wellknown_count in [1,2,4,10]:
        #     for forget_count in [1,2,4]:
        #         inputs += [get_format_string(data_point['wellknown_question'][:wellknown_count] + data_point['forget_question'][:forget_count])]
        
        sanity_generation = []
        for inp in tqdm(inputs):
            # print(inp)
            encoded_inputs = tokenizer(inp, padding=True, truncation=True, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = model.generate(**encoded_inputs, max_new_tokens=768, do_sample=False, temperature=0.0).detach().cpu()
                # sanity_generation = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                sanity_generation.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

        sanity_generation = [generation.split("[ANSWER] ")[1].strip() for generation in sanity_generation]

        with open(f"data/tofu/casestudy_{method_name}_{lr}.jsonl", 'a') as writer:
            idx_count = 0
            new_line = deepcopy(data_point)
            # for forget_count in [1,2,4]:
            #     new_line[f"forget{forget_count}_forget0"] = sanity_generation[idx_count]
            #     idx_count += 1
            # for retain_count in [1,2,4,10]:
            #     new_line[f"retain{retain_count}_retain0"] = sanity_generation[idx_count]
            #     idx_count += 1
            for forget_count in [1,2,4]:
                for retain_count in [1,2,4]:
                    new_line[f"forget{forget_count}_retain{retain_count}"] = sanity_generation[idx_count]
                    idx_count += 1
            # for forget_count in [1,2,4]:
            #     for wellknown_count in [1,2,4,10]:
            #         new_line[f"forget{forget_count}_wellknown{wellknown_count}"] = sanity_generation[idx_count]
            #         idx_count += 1
            for retain_count in [1,2,4]:
                for forget_count in [1,2,4]:
                    new_line[f"retain{retain_count}_forget{forget_count}"] = sanity_generation[idx_count]
                    idx_count += 1
            # for wellknown_count in [1,2,4,10]:
            #     for forget_count in [1,2,4]:
            #         new_line[f"wellknown{wellknown_count}_forget{forget_count}"] = sanity_generation[idx_count]
            #         idx_count += 1
            writer.write(json.dumps(new_line) + '\n')
        
def load_jsonl(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                data.append(json.loads(line))
    return data

if __name__ == "__main__":

    # methods = ["DPO+GD", "DPO+KL", "GA+KL"]
    # lrs = [1e-5, 1e-5, 1e-5]
    # epochs = [15, 15, 15]

    # methods = ["GA+GD", "IDK+GD", "IDK+KL"]
    # lrs = [1e-5, 1e-5, 1e-5]
    # epochs = [15, 10, 10]

    # methods = ["NPO+GD", "IDK+AP", "IDK+AP+NM_JWJ"]
    # lrs = [5e-5, 1e-5, 1e-5]
    # epochs = [10, 10, 10]

    # methods = ["NPO+KL", "IDK+AP+reverse_mixed_JWJ", "IDK+AP+mixed_JWJ"]
    # lrs = [5e-5, 1e-5, 1e-5]
    # epochs = [10, 10, 10]

    # methods = ["IDK+AP+JWJ"]
    # lrs = [1e-5]
    # epochs = [10]

    methods = ["ME+GD"]
    lrs = [1e-5]
    epochs = [10]

    for method, lr, epoch in zip(methods, lrs, epochs):
        print(f"Generate {method} batch with LR {lr} in epoch {epoch}")
        main(method, 'forget01', lr, epoch)
