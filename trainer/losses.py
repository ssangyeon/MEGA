import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from transformers import AutoTokenizer

def get_loss(model, ref_model, inputs, loss_type, beta=0.1):
    # forget_loss
    if 'GA' in loss_type:
        forget_loss = ga_loss(model, inputs)
    elif 'NPO' in loss_type:
        forget_loss = npo_loss(model, ref_model, inputs, beta=beta)
    elif 'DPO' in loss_type:
        forget_loss = dpo_loss(model, ref_model, inputs, beta=beta)
    elif 'ME' in loss_type:
        forget_loss = me_loss(model, inputs)
    elif 'PIDK' in loss_type:
        forget_loss = pidk_loss(model, inputs)
    elif 'IDK' in loss_type:
        forget_loss = idk_loss(model, inputs)
    elif 'SY' in loss_type:
        forget_loss = sy_loss(model, inputs)
    else: forget_loss = 0


    # regularization_loss
    if 'GD' in loss_type:
        regularization_loss = gd_loss(model, inputs)
    elif 'KL' in loss_type:
        regularization_loss = kl_loss(model, ref_model, inputs)
    elif 'PPAP' in loss_type:    
        regularization_loss = ppap_loss(model, inputs, beta=beta)
    elif 'PAP' in loss_type:
        regularization_loss = pap_loss(model, inputs, beta=beta)
    elif 'AP' in loss_type:
        regularization_loss = ap_loss(model, inputs, beta=beta)
    else:
        regularization_loss = 0

    if loss_type == 'LLMU':
        forget_loss = ga_loss(model, inputs)
        regularization_loss = mismatch_loss(model, inputs) + kl_loss(model, ref_model, inputs)
    elif 'MK' in loss_type:
        match = re.search(r'ALBERT([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        regularization_loss += number * ((mixed_fr_untargeted_kl_loss(model, ref_model, inputs) + mixed_rf_untargeted_kl_loss(model, ref_model, inputs)))
    elif 'MG' in loss_type:
        match = re.search(r'ALBERT([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        regularization_loss += number * (mixed_fr_untargeted_gd_loss(model, ref_model, inputs) + mixed_rf_untargeted_gd_loss(model, ref_model, inputs))
    elif 'MTK' in loss_type:
        match = re.search(r'ALBERT([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        regularization_loss += number * (mixed_fr_targeted_kl_loss(model, ref_model, inputs) + mixed_rf_targeted_kl_loss(model, ref_model, inputs))
    elif 'UGAD' in loss_type:
        regularization_loss += (fr_ugad_loss(model, inputs) + rf_ugad_loss(model, inputs))

    #########################################################################################################
        ##1. fr: PO + rf: gd_rf_nm_loss
        ##2. fr: PO + rf: gd_rf_loss
        ##3. fr: PO + rf: ref사용x | 분모:(A_r)+IDK. 분자:(A_r)+A_f
        ##4. fr: PO + rf: ref사용o | 분모:(A_r)+IDK, 분자:(A_r)+A_f|ref
        ##5. fr: PO + rf: ref사용o | 분모:A_r + IDK, 분자: A_r + A_f|ref
        ##6. fr: gd_fr_nm_loss + rf: ref사용x | 분모:(A_r)+IDK. 분자:(A_r)+A_f
        ##7. fr: gd_fr_nm_loss + rf: ref사용o | 분모:(A_r)+IDK, 분자:(A_r)+A_f|ref
        ##8. fr: gd_fr_nm_loss + rf: ref사용o | 분모:A_r + IDK, 분자: A_r + A_f|ref
        ##목표: 분자 확률 줄이기/ 분모 확률 올리기 // 분모>>분자일떄 학습x
        ## fr: ref사용o | 분모: (IDK)+A_r, 분자:(A_f)+A_r|ref
        ## IDK+AP+1ALBERT
        ## IDK+AP+2ALBERT
        ## IDK+AP+3ALBERT
        ## IDK+AP+4ALBERT
        ## IDK+AP+5ALBERT
        ## IDK+AP+1ALBERT0.5
        ## IDK+AP+2ALBERT0.5
        ## IDK+AP+3ALBERT0.5
        ## IDK+AP+4ALBERT0.5
        ## IDK+AP+5ALBERT0.5

    if '1ALBERT' in loss_type:  
        match = re.search(r'ALBERT([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        regularization_loss += 2 * number * albert_fr_loss(model, ref_model, inputs, beta=beta)
        regularization_loss += gd_rf_nm_loss(model, inputs)
    elif '2ALBERT' in loss_type:  
        match = re.search(r'ALBERT([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        regularization_loss += 2 * number * albert_fr_loss(model, ref_model, inputs, beta=beta)
        regularization_loss += gd_rf_loss(model, inputs)
    elif '3ALBERT' in loss_type:  ##3. fr: PO + rf: ref사용x | 분모:(A_r)+IDK. 분자:(A_r)+A_f
        match = re.search(r'ALBERT([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        regularization_loss += 2 * number * albert_fr_loss(model, ref_model, inputs, beta=beta)
        regularization_loss += number * albert_rf_loss3(model, ref_model, inputs, beta=beta)
    elif '4ALBERT' in loss_type:  ##4. fr: PO + rf: ref사용o | 분모:(A_r)+IDK, 분자:(A_r)+A_f|ref
        match = re.search(r'ALBERT([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        regularization_loss += 2 * number * albert_fr_loss(model, ref_model, inputs, beta=beta)
        regularization_loss += 2 * number * albert_rf_loss4(model, ref_model, inputs, beta=beta)
    elif '5ALBERT' in loss_type:  ##5. fr: PO + rf: ref사용o | 분모:A_r + IDK, 분자: A_r + A_f|ref
        match = re.search(r'ALBERT([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        regularization_loss += 2 * number * albert_fr_loss(model, ref_model, inputs, beta=beta)
        regularization_loss += 2 * number * albert_rf_loss5(model, ref_model, inputs, beta=beta)
    elif '_6ALBERT' in loss_type:  
        match = re.search(r'ALBERT([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        regularization_loss += 2 * number * albert_fr_loss(model, ref_model, inputs, beta=beta)
        regularization_loss += 2 * number * albert_rf_dpo_loss(model, ref_model, inputs, beta=beta)
    elif '6ALBERT' in loss_type:  
        match = re.search(r'ALBERT([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        regularization_loss += 2 * number * albert_fr_loss(model, ref_model, inputs, beta=beta)
        regularization_loss += number * albert_rf_dpo_loss(model, ref_model, inputs, beta=beta)
    elif '_7ALBERT' in loss_type:  
        match = re.search(r'ALBERT([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        regularization_loss += gd_fr_untargeted_loss(model, inputs)
        regularization_loss += number * albert_rf_loss6(model, ref_model, inputs, beta=beta)
    elif '7ALBERT' in loss_type:  
        match = re.search(r'ALBERT([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        regularization_loss += gd_fr_untargeted_loss(model, inputs)
        regularization_loss += number * albert_rf_loss6(model, ref_model, inputs, beta=beta)
    
    #########################################################################################################

    if 'reverse_mixed_untargeted_JWJ' in loss_type:
        match = re.search(r'JWJ([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        fr_loss = gd_fr_untargeted_loss(model, inputs)
        rf_loss = gd_rf_nm_untargeted_loss(model, inputs)
        regularization_loss += number * (fr_loss + rf_loss) 
    elif 'half_NM_untargeted_JWJ' in loss_type:
        match = re.search(r'JWJ([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        fr_loss = 0
        rf_loss = gd_rf_nm_untargeted_loss(model, inputs)
        regularization_loss += number * (fr_loss + rf_loss) 
    elif 'NM_untargeted_JWJ' in loss_type:
        match = re.search(r'JWJ([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        fr_loss = gd_fr_nm_untargeted_loss(model, inputs)
        rf_loss = gd_rf_nm_untargeted_loss(model, inputs)
        regularization_loss += number * (fr_loss + rf_loss)     
    elif 'half_untargeted_JWJ' in loss_type:
        match = re.search(r'JWJ([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        fr_loss = 0
        rf_loss = gd_rf_untargeted_loss(model, inputs)
        regularization_loss += number * (fr_loss + rf_loss) 
    elif 'mixed_untargeted_JWJ' in loss_type:
        match = re.search(r'JWJ([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        fr_loss = gd_fr_nm_untargeted_loss(model, inputs)
        rf_loss = gd_rf_untargeted_loss(model, inputs)
        regularization_loss += number * (fr_loss + rf_loss)     
    elif 'untargeted_JWJ' in loss_type:
        match = re.search(r'JWJ([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        fr_loss = gd_fr_untargeted_loss(model, inputs)
        rf_loss = gd_rf_untargeted_loss(model, inputs)
        regularization_loss += number * (fr_loss + rf_loss)
    elif "NM_JWJ" in loss_type:
        match = re.search(r'JWJ([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        fr_loss = gd_fr_nm_loss(model, inputs)
        rf_loss = gd_rf_nm_loss(model, inputs)
        regularization_loss += number * (fr_loss + rf_loss)
    elif "reverse_mixed_JWJ" in loss_type:
        match = re.search(r'JWJ([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        fr_loss = gd_fr_loss(model, inputs)
        rf_loss = gd_rf_nm_loss(model, inputs)
        regularization_loss += number * (fr_loss + rf_loss)    
    elif "mixed_JWJ" in loss_type:
        match = re.search(r'JWJ([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        fr_loss = gd_fr_nm_loss(model, inputs)
        rf_loss = gd_rf_loss(model, inputs)
        regularization_loss += number * (fr_loss + rf_loss)    
    elif 'JWJ' in loss_type:
        match = re.search(r'JWJ([\d.]+)', loss_type)
        if match:
            number = float(match.group(1))  # 숫자를 float으로 변환
        else:
            number = 1.0
        fr_loss = gd_fr_loss(model, inputs)
        rf_loss = gd_rf_loss(model, inputs)
        regularization_loss += number * (fr_loss + rf_loss) 
    

    return forget_loss, regularization_loss

    


def ga_loss(model, inputs):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss = -1 * outputs.loss
    return loss


def npo_loss(model, ref_model, inputs, beta=0.1):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss_current = get_batch_loss(outputs.logits, labels)

    with torch.no_grad():
        ref_outputs = ref_model(input_ids, labels=labels,
                                attention_mask=attention_mask)
        loss_ref = get_batch_loss(ref_outputs.logits, labels)

    neg_log_ratios = loss_current - loss_ref
    loss = - F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta

    return loss



def idk_loss(model, inputs):
    forget_idk_inputs = inputs[2]
    input_ids, labels, attention_mask = forget_idk_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss

def sy_loss(model, inputs):
    forget_idk_inputs = inputs[8]
    input_ids, labels, attention_mask = forget_idk_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss    

def pidk_loss(model, inputs):
    forget_idk_inputs = inputs[5]
    input_ids, labels, attention_mask = forget_idk_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss

def dpo_loss(model, ref_model, inputs, beta=0.1):
    forget_inputs, forget_idk_inputs = inputs[0], inputs[2]
    forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
    idk_input_ids, idk_labels, idk_attention_mask = forget_idk_inputs

    idk_outputs = model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
    forget_outputs = model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
    idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
    forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)

    with torch.no_grad():
        idk_outputs_ref = ref_model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
        forget_outputs_ref = ref_model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
        idk_loss_ref = -1 * get_batch_loss(idk_outputs_ref.logits, idk_labels)
        forget_loss_ref = -1 * get_batch_loss(forget_outputs_ref.logits, forget_labels)

    pi_logratios = idk_loss_current - forget_loss_current
    ref_logratios = idk_loss_ref - forget_loss_ref
    loss = - F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean() * 2 / beta
    return loss


# Regularization Loss: AP
def ap_loss(model, inputs, beta=0.1):
    retain_inputs, retain_idk_inputs = inputs[1], inputs[3]
    retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
    retain_idk_input_ids, retain_idk_labels, retain_idk_attention_mask = retain_idk_inputs

    outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
    idk_outputs = model(retain_idk_input_ids, labels=retain_idk_labels, attention_mask=retain_idk_attention_mask)

    loss = get_batch_loss(outputs.logits, retain_labels)
    loss_idk = get_batch_loss(idk_outputs.logits, retain_idk_labels)

    neg_log_ratios = loss_idk - loss

    loss = - F.logsigmoid(beta * neg_log_ratios).mean() / beta

    return loss

def pap_loss(model, inputs, beta=0.1):
    retain_inputs, retain_idk_inputs = inputs[6], inputs[3]
    retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
    retain_idk_input_ids, retain_idk_labels, retain_idk_attention_mask = retain_idk_inputs

    outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
    idk_outputs = model(retain_idk_input_ids, labels=retain_idk_labels, attention_mask=retain_idk_attention_mask)

    loss = get_batch_loss(outputs.logits, retain_labels)
    loss_idk = get_batch_loss(idk_outputs.logits, retain_idk_labels)

    neg_log_ratios = loss_idk - loss

    loss = - F.logsigmoid(beta * neg_log_ratios).mean() / beta

    return loss


def ppap_loss(model, inputs, beta=0.1):
    retain_inputs, retain_idk_inputs = inputs[6], inputs[7]
    retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
    retain_idk_input_ids, retain_idk_labels, retain_idk_attention_mask = retain_idk_inputs

    outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
    idk_outputs = model(retain_idk_input_ids, labels=retain_idk_labels, attention_mask=retain_idk_attention_mask)

    loss = get_batch_loss(outputs.logits, retain_labels)
    loss_idk = get_batch_loss(idk_outputs.logits, retain_idk_labels)

    neg_log_ratios = loss_idk - loss

    loss = - F.logsigmoid(beta * neg_log_ratios).mean() / beta

    return loss

# Regularization Loss: KL
def kl_loss(model, ref_model, inputs):
    retain_inputs = inputs[1]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    probs = F.log_softmax(outputs.logits, dim=-1).view(-1, outputs.logits.shape[-1])

    with torch.no_grad():
        outputs_ref = ref_model(input_ids, labels=labels, attention_mask=attention_mask)
    ref_probs = F.log_softmax(outputs_ref.logits, dim=-1).view(-1, outputs_ref.logits.shape[-1])

    loss = nn.functional.kl_div(
        probs, ref_probs, reduction='batchmean', log_target=True)

    return loss


def mismatch_loss(model, inputs):
    mismatch_inputs = inputs[4]
    input_ids, labels, attention_mask = mismatch_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)

    loss = outputs.loss
    return loss


# Regularization Loss: GD
def gd_loss(model, inputs):
    retain_inputs = inputs[1]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss

def gd_fr_loss(model, inputs):
    retain_inputs = inputs[9]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss

def gd_rf_loss(model, inputs):
    retain_inputs = inputs[10]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss

def gd_fr_untargeted_loss(model, inputs):
    retain_inputs = inputs[13]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss

def gd_rf_untargeted_loss(model, inputs):
    retain_inputs = inputs[14]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = -1 * outputs.loss
    return loss

def albert_rf_loss(model, ref_model, inputs, beta=0.1):
    forget_inputs = inputs[14]
    input_ids, labels, attention_mask = forget_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss_current = get_batch_loss(outputs.logits, labels)

    with torch.no_grad():
        ref_outputs = ref_model(input_ids, labels=labels,
                                attention_mask=attention_mask)
        loss_ref = get_batch_loss(ref_outputs.logits, labels)

   
    neg_log_ratios = loss_current - loss_ref 
    loss = - F.logsigmoid(beta * neg_log_ratios).mean() / beta

    return loss

######################################################################
def albert_rf_loss3(model, ref_model, inputs, beta=0.1):##3. rf: ref사용x | 분모:(A_r)+IDK:(10). 분자:(A_r)+A_f:(14) : 분자 - 분모
    forget_inputs1 = inputs[10] ##분모
    input_ids1, labels1, attention_mask1 = forget_inputs1

    outputs1 = model(input_ids1, labels=labels1,
                    attention_mask=attention_mask1)
    loss_current1 = get_batch_loss(outputs1.logits, labels1)

    forget_inputs2 = inputs[14] ##분자
    input_ids2, labels2, attention_mask2 = forget_inputs2

    outputs2 = model(input_ids2, labels=labels2,
                    attention_mask=attention_mask2)
    loss_current2 = get_batch_loss(outputs2.logits, labels2)
    
    neg_log_ratios = loss_current2 - loss_current1  
    loss = - F.logsigmoid(beta * neg_log_ratios).mean() / beta

    return loss
def albert_rf_loss4(model, ref_model, inputs, beta=0.1):##4.rf: ref사용o | 분모:(A_r)+IDK:(10), 분자:(A_r)+A_f|ref : (14)
    forget_inputs1 = inputs[10] ##분모
    input_ids1, labels1, attention_mask1 = forget_inputs1

    outputs1 = model(input_ids1, labels=labels1,
                    attention_mask=attention_mask1)
    loss_current = get_batch_loss(outputs1.logits, labels1)

    forget_inputs2 = inputs[14] ##분자
    input_ids2, labels2, attention_mask2 = forget_inputs2
    with torch.no_grad():
        ref_outputs = ref_model(input_ids2, labels=labels2,
                                attention_mask=attention_mask2)
        loss_ref = get_batch_loss(ref_outputs.logits, labels2)

    neg_log_ratios = loss_ref - loss_current  
    loss = - F.logsigmoid(beta * neg_log_ratios).mean() / beta

    return loss
def albert_rf_loss5(model, ref_model, inputs, beta=0.1):##5. fr: rf: ref사용o | 분모:A_r + IDK:(10), 분자: A_r + A_f|ref : (16)
    forget_inputs1 = inputs[10] ##분모
    input_ids1, labels1, attention_mask1 = forget_inputs1

    outputs1 = model(input_ids1, labels=labels1,
                    attention_mask=attention_mask1)
    loss_current = get_batch_loss(outputs1.logits, labels1)

    forget_inputs2 = inputs[16] ##분자
    input_ids2, labels2, attention_mask2 = forget_inputs2
    with torch.no_grad():
        ref_outputs = ref_model(input_ids2, labels=labels2,
                                attention_mask=attention_mask2)
        loss_ref = get_batch_loss(ref_outputs.logits, labels2)

    neg_log_ratios = loss_ref - loss_current 
    loss = - F.logsigmoid(beta * neg_log_ratios).mean() / beta

    return loss

def albert_rf_dpo_loss(model, ref_model, inputs, beta=0.1):
    forget_inputs, forget_idk_inputs = inputs[14], inputs[10]
    forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
    idk_input_ids, idk_labels, idk_attention_mask = forget_idk_inputs

    idk_outputs = model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
    forget_outputs = model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
    idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
    forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)

    with torch.no_grad():
        idk_outputs_ref = ref_model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
        forget_outputs_ref = ref_model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
        idk_loss_ref = -1 * get_batch_loss(idk_outputs_ref.logits, idk_labels)
        forget_loss_ref = -1 * get_batch_loss(forget_outputs_ref.logits, forget_labels)

    pi_logratios = idk_loss_current - forget_loss_current
    ref_logratios = idk_loss_ref - forget_loss_ref
    loss = - F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean() / beta
    return loss

def albert_rf_half_dpo_loss(model, ref_model, inputs, beta=0.1):
    forget_inputs, forget_idk_inputs = inputs[14], inputs[12]
    forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
    idk_input_ids, idk_labels, idk_attention_mask = forget_idk_inputs

    idk_outputs = model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
    forget_outputs = model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
    idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
    forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)

    with torch.no_grad():
        idk_outputs_ref = ref_model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
        forget_outputs_ref = ref_model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
        idk_loss_ref = -1 * get_batch_loss(idk_outputs_ref.logits, idk_labels)
        forget_loss_ref = -1 * get_batch_loss(forget_outputs_ref.logits, forget_labels)

    pi_logratios = idk_loss_current - forget_loss_current
    ref_logratios = idk_loss_ref - forget_loss_ref
    loss = - F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean() / beta
    return loss

def albert_rf_loss6(model, ref_model, inputs, beta=0.1):
    forget_inputs = inputs[14]
    input_ids, labels, attention_mask = forget_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss_current = get_batch_loss(outputs.logits, labels)

    with torch.no_grad():
        ref_outputs = ref_model(input_ids, labels=labels,
                                attention_mask=attention_mask)
        loss_ref = get_batch_loss(ref_outputs.logits, labels)

    neg_log_ratios = loss_current - loss_ref
    loss = - F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta

    return loss
######################################################################
def albert_fr_loss(model, ref_model, inputs, beta=0.1):
    forget_inputs = inputs[9]
    input_ids, labels, attention_mask = forget_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss_current = get_batch_loss(outputs.logits, labels)

    forget_inputs2 = inputs[13]
    input_ids2, labels2, attention_mask2 = forget_inputs2

    with torch.no_grad():
        ref_outputs = ref_model(input_ids2, labels=labels2,
                                attention_mask=attention_mask2)
        loss_ref = get_batch_loss(ref_outputs.logits, labels2)

    neg_log_ratios = loss_ref - loss_current
    loss = - F.logsigmoid(beta * neg_log_ratios).mean() / beta

    return loss

def gd_fr_nm_untargeted_loss(model, inputs):
    retain_inputs = inputs[15]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss

def gd_rf_nm_untargeted_loss(model, inputs):
    retain_inputs = inputs[16]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = -1 * outputs.loss
    return loss

def gd_fr_nm_loss(model, inputs):
    retain_inputs = inputs[11]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss



def gd_rf_nm_loss(model, inputs):
    retain_inputs = inputs[12]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss

def get_batch_loss(logits, labels):
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss


def me_loss(model, inputs):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=None, attention_mask=attention_mask)
    loss = get_me_loss(outputs.logits, labels)

    return loss


def get_me_loss(logits, labels):
    num_labels = logits.shape[-1]

    assert logits.shape[:-1] == labels.shape, "Logits and labels must have compatible shapes."

    # Adjust logits and labels to exclude the last token
    labels = labels[:, 1:].clone()  # (bs, seq_len - 1)
    logits = logits[:, :-1, :]  # (bs, seq_len - 1, vocab_size)

    soft_outputs = F.softmax(logits, dim=-1).view(-1, num_labels)  # (bs*seq_len, vocab_size)
    uniform_dist = torch.full_like(soft_outputs, 1.0 / num_labels).to(logits.device)  # (bs*seq_len, vocab_size)

    loss_mask = (labels != -100).view(-1)  # (bs*(seq_len - 1))

    kl_div = F.kl_div((soft_outputs + 1e-12).log(), uniform_dist, reduction='none').sum(-1)  # (bs*(seq_len - 1))

    masked_kl_div = kl_div * loss_mask  # (bs*(seq_len - 1))
    loss = masked_kl_div.sum() / loss_mask.sum()

    return loss

#############################################################################



def find_all_token_indices_batch1(input_ids):
    """
    input_ids: (batch_size, seq_len)
    tokenizer: Hugging Face 토크나이저 객체
    target_strings: 찾고자 하는 문자열 리스트
    """
    target_strings = ["[INST] "," [/INST]", "\n", "1."] ##fr : 이 범위 내에서만 uniform
    # target_strings = ["2.", "[/INST]"] ##rf:  이 범위 내에서만 uniform
    batch_indices = []  # 각 배치별 결과 저장
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    # 각 배치별로 처리
    for batch_idx, batch in enumerate(input_ids):  # 배치 인덱스를 포함하여 반복
        batch_result = {}
        # print(f"\n=== Debugging Batch FR ===")
        # print(f"Input IDs: {batch.tolist()}")  # 배치의 전체 input_ids 출력


        for j, target in enumerate(target_strings):
            # 1. 타겟 문자열을 토크나이즈하여 토큰 ID로 변환
            target_token_ids = tokenizer(target, add_special_tokens=False)["input_ids"][1:]
            # print(f"\nTarget String: '{target}'")
            # print(f"Target Token IDs: {target_token_ids}")

            # 2. input_ids에서 대상 문자열의 모든 토큰 ID 위치 찾기
            target_indices = []
            for i in range(len(batch) - len(target_token_ids) + 1):
                # 현재 비교 중인 범위 출력
                if batch[i:i+len(target_token_ids)].tolist() == target_token_ids:
                    target_indices.append(i)  # 시작 인덱스를 추가
            # print(f"Matched Indices for '{target}': {target_indices}")

            # 결과 저장 (해당 타겟 문자열에 대한 결과)
            batch_result[target] = target_indices if target_indices else -1  # 없으면 -1 저장

        # 각 배치의 결과 저장
        batch_indices.append(batch_result)
        # print(f"\nBatch Result: {batch_result}")

    return batch_indices
def get_mixed_untargeted_loss1(logits, labels, input_ids, ref_logits):
    """
    logits: 메인 모델의 출력 (bs, seq_len, vocab_size)
    labels: 정답 라벨 (bs, seq_len)
    input_ids: 입력 토큰 ID (bs, seq_len)
    ref_logits: 기준 모델의 출력 (bs, seq_len, vocab_size)
    """
    # 타겟 문자열들의 모든 토큰 인덱스 찾기
    indices = find_all_token_indices_batch1(input_ids)
    num_labels = logits.shape[-1]

    # Adjust logits and labels to exclude the last token
    labels = labels[:, 1:].clone()  # (bs, seq_len - 1)
    logits = logits[:, :-1, :]  # (bs, seq_len - 1, vocab_size)
    ref_logits = ref_logits[:, :-1, :]  # 기준 모델도 동일하게 조정


    # Flatten logits for processing
    batch_size, seq_len_minus_one = labels.shape
    flat_logits = logits.reshape(-1, num_labels)  # `view` 대신 `reshape` 사용
    flat_ref_logits = ref_logits.reshape(-1, num_labels)  # 동일한 변경 적용

    # 기준 모델에 유니폼 분포 강제
    uniform_dist = torch.full_like(flat_ref_logits, 1.0 / num_labels)

    for batch_idx in range(batch_size):
        ranges = []
        batch_indices = indices[batch_idx]
        if " [/INST]" in batch_indices and "\n" in batch_indices:
            one_positions = batch_indices[" [/INST]"]
            newline_positions = batch_indices["\n"]
            start_positions = batch_indices["[INST] "]
            for i, pos in enumerate(start_positions):
                if i==0:
                    ranges.append((pos, newline_positions[0]))
            for i, pos in enumerate(one_positions):
                if i == 0:  
                    ranges.append((pos, newline_positions[1]))

        # Apply uniform distribution to specific ranges in ref_logits
        # print(f"rangefr:{ranges}")
        for i, (start, end) in enumerate(ranges):  # ranges는 [(50, 118)]
            if i==0:
                # print(f"Applying uniform distribution from {start} to {end}")
                for idx in range(start+5, end):  # 범위에 포함된 모든 토큰 처리
                    flat_idx = batch_idx * seq_len_minus_one + idx
                    if flat_idx < flat_ref_logits.size(0):  # 유효한 flat_idx인지 확인
                        flat_ref_logits[flat_idx] = uniform_dist[flat_idx]
                    else:
                        print(f"Warning: flat_idx {flat_idx} is out of range")
            elif i==1:
                # print(f"Applying uniform distribution from {start} to {end}")
                for idx in range(start+6, end):  # 범위에 포함된 모든 토큰 처리
                    flat_idx = batch_idx * seq_len_minus_one + idx
                    if flat_idx < flat_ref_logits.size(0):  # 유효한 flat_idx인지 확인
                        flat_ref_logits[flat_idx] = uniform_dist[flat_idx]
                    else:
                        print(f"Warning: flat_idx {flat_idx} is out of range")

# ############################################################################################################
    # # # Compute KL divergence between main model and reference model
    # print("\n=== Debug: FR===")
    # print(f"flat_logits shape: {flat_logits.shape}, flat_ref_logits shape: {flat_ref_logits.shape}")
    # # 배치 크기 제한 (첫 2개 배치만 확인)
    # batch_size = logits.size(0)
    # seq_len_minus_one = logits.size(1)
    # print(f"batch_size: {batch_size}")
    # print(f"seq_len_minus_one: {seq_len_minus_one}")
    # for batch_idx in range(min(batch_size, 2)):  # 첫 2개 배치만 출력
    #     print(f"\n[Batch {batch_idx}]")
    #     for seq_idx in range(30,min(seq_len_minus_one, 130)):  # 각 배치의 첫 3개 토큰
    #         flat_idx = batch_idx * seq_len_minus_one + seq_idx
    #         print(f"  Token {seq_idx}:")
    #         print(f"    flat_logits[:5]: {flat_logits[flat_idx][:5].tolist()}")
    #         print(f"    flat_ref_logits[:5]: {flat_ref_logits[flat_idx][:5].tolist()}")
# ############################################################################################################
    # Compute KL divergence between main model and reference model
    # kl_div = F.kl_div(
    #     (flat_logits + 1e-12).log(), F.log_softmax(flat_ref_logits, dim=-1), reduction="batchmean",log_target=True
    # )
    # Softmax로 확률 분포 계산
    soft_outputs = F.softmax(flat_logits, dim=-1)  # 확률 분포 (1996, 32000)
    soft_ref_outputs = F.softmax(flat_ref_logits, dim=-1)  # 기준 확률 분포 (1996, 32000)

    # Softmax 값 클리핑 (수치 안정성 확보)
    soft_outputs = torch.clamp(soft_outputs, min=1e-12, max=1.0)
    soft_ref_outputs = torch.clamp(soft_ref_outputs, min=1e-12, max=1.0)

    # KL-divergence 계산
    kl_div = (soft_outputs * (torch.log(soft_outputs) - torch.log(soft_ref_outputs))).sum(-1)

    loss_mask = (labels != -100).view(-1)
    masked_kl_div = kl_div *loss_mask

    # 전체 평균 KL-divergence
    loss = masked_kl_div.sum() / loss_mask.sum()
    # print(f"kl_div:{kl_div}")
    # print(f"masked_kl_div:{masked_kl_div[20:70]}")
    # print(f"loss: {loss}")

    return loss


def mixed_fr_untargeted_kl_loss(model, ref_model, inputs):
    retain_inputs = inputs[15]  ##forget+retain_nomask_untargeted
    input_ids, labels, attention_mask = retain_inputs

    # Main model outputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

    # Reference model outputs
    with torch.no_grad():
        outputs_ref = ref_model(input_ids, labels=labels, attention_mask=attention_mask)

    # Compute KL divergence between main and reference model
    loss = get_mixed_untargeted_loss1(
        outputs.logits, labels, input_ids, ref_logits=outputs_ref.logits
    )

    return loss

def mixed_rf_untargeted_kl_loss(model, ref_model, inputs):
    retain_inputs = inputs[16]  ##retain+forget_nomask_untargeted
    input_ids, labels, attention_mask = retain_inputs

    # Main model outputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

    # Reference model outputs
    with torch.no_grad():
        outputs_ref = ref_model(input_ids, labels=labels, attention_mask=attention_mask)

    # Compute KL divergence between main and reference model
    loss = get_mixed_untargeted_loss2(
        outputs.logits, labels, input_ids, attention_mask, ref_logits=outputs_ref.logits
    )

    return loss

def get_mixed_untargeted_loss2(logits, labels, input_ids, attention_mask, ref_logits):
    """
    logits: 메인 모델의 출력 (bs, seq_len, vocab_size)
    labels: 정답 라벨 (bs, seq_len)
    input_ids: 입력 토큰 ID (bs, seq_len)
    ref_logits: 기준 모델의 출력 (bs, seq_len, vocab_size)
    """
    # 타겟 문자열들의 모든 토큰 인덱스 찾기
    indices = find_all_token_indices_batch2(input_ids, attention_mask)
    num_labels = logits.shape[-1]

    # Adjust logits and labels to exclude the last token
    labels = labels[:, 1:].clone()  # (bs, seq_len - 1)
    logits = logits[:, :-1, :]  # (bs, seq_len - 1, vocab_size)
    ref_logits = ref_logits[:, :-1, :]  # 기준 모델도 동일하게 조정

    # Flatten logits for processing
    batch_size, seq_len_minus_one = labels.shape
    flat_logits = logits.reshape(-1, num_labels)  # `view` 대신 `reshape` 사용
    flat_ref_logits = ref_logits.reshape(-1, num_labels)

    # 기준 모델에 유니폼 분포 강제
    uniform_dist = torch.full_like(flat_ref_logits, 1.0 / num_labels)

    for batch_idx in range(batch_size):
        ranges = []
        batch_indices = indices[batch_idx]
        # print(batch_indices)
        # if "\n" in batch_indices:
        #     sub_dict = batch_indices["\n"]

        #     if "second_2_start" in sub_dict and "valid_end" in sub_dict:
        #         one_position = sub_dict["second_2_start"]
        #         newline_position = sub_dict["valid_end"]

        #         ranges.append((one_position, newline_position))
        one_positions = batch_indices["\n"]
        newline_positions1 = batch_indices[" [/INST]"]
        newline_positions2 = batch_indices["valid_end"]
        for i, pos in enumerate(one_positions):
            if i ==0:
                ranges.append((pos,newline_positions1[0]))
            elif i==1:
                ranges.append((pos,newline_positions2))
        
        # Apply uniform distribution to specific ranges in ref_logits
        # print(f"rangerf:{ranges}")
        for start, end in ranges:
            for idx in range(start+3, end):
                flat_idx = batch_idx * seq_len_minus_one + idx
                flat_ref_logits[flat_idx] = uniform_dist[flat_idx]
# ############################################################################################################
    # # Compute KL divergence between main model and reference model
    # print("\n=== Debug:RF ===")
    # print(f"flat_logits shape: {flat_logits.shape}, flat_ref_logits shape: {flat_ref_logits.shape}")
    # print(ranges)
    # # 배치 크기 제한 (첫 2개 배치만 확인)
    # batch_size = logits.size(0)
    # seq_len_minus_one = logits.size(1)
    # num_labels = logits.size(-1)

    # for batch_idx in range(min(batch_size, 2)):  # 첫 2개 배치만 출력
    #     print(f"\n[Batch {batch_idx}]")
    #     for seq_idx in range(50,min(seq_len_minus_one, 100)):  # 각 배치의 첫 3개 토큰
    #         flat_idx = batch_idx * seq_len_minus_one + seq_idx
    #         print(f"  Token {seq_idx}:")
    #         print(f"    flat_logits[:5]: {flat_logits[flat_idx][:5].tolist()}")
    #         print(f"    flat_ref_logits[:5]: {flat_ref_logits[flat_idx][:5].tolist()}")
# ############################################################################################################
    # Softmax로 확률 분포 계산
    soft_outputs = F.softmax(flat_logits, dim=-1)  # 확률 분포 (1996, 32000)
    soft_ref_outputs = F.softmax(flat_ref_logits, dim=-1)  # 기준 확률 분포 (1996, 32000)

    # Softmax 값 클리핑 (수치 안정성 확보)
    soft_outputs = torch.clamp(soft_outputs, min=1e-12, max=1.0)
    soft_ref_outputs = torch.clamp(soft_ref_outputs, min=1e-12, max=1.0)

    # KL-divergence 계산
    kl_div = (soft_outputs * (torch.log(soft_outputs) - torch.log(soft_ref_outputs))).sum(-1)

    loss_mask = (labels != -100).view(-1)
    masked_kl_div = kl_div *loss_mask

    # 전체 평균 KL-divergence
    loss = masked_kl_div.sum() / loss_mask.sum()
    # print(f"kl_div:{kl_div}")
    # print(f"masked_kl_div:{masked_kl_div[20:70]}")
    # print(f"loss: {loss}")

    return loss


def find_all_token_indices_batch2(input_ids, attention_mask):
    """
    input_ids: (batch_size, seq_len) - 모델에 들어가는 입력 ID
    attention_mask: (batch_size, seq_len) - 유효 토큰(1)과 패딩(0) 구분
    pad_token_id: 패딩 토큰 ID (default: -100)
    """
    target_strings = ["\n", " [/INST]"]  # 두 번째 "2."의 시작 위치 탐지
    batch_indices = []  # 각 배치별 결과 저장
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    pad_token_id = tokenizer.eos_token_id
    for batch_idx, batch in enumerate(input_ids):
        batch_result = {}
        # print(f"\n=== Debugging Batch {batch_idx} ===")
        # print(f"Input IDs: {batch.tolist()}")

        # 유효한 문장의 끝 탐지
        if attention_mask is not None:
            valid_end = attention_mask[batch_idx].nonzero(as_tuple=True)[0][-1].item() + 1
            # print(f"Valid end (using attention mask): {valid_end}")
        else:
            valid_end = (batch != pad_token_id).nonzero(as_tuple=True)[0][-1].item() + 1
            # print(f"Valid end (using pad token): {valid_end}")
        batch_result["valid_end"] = valid_end 
        for i, target in enumerate(target_strings):
            # 타겟 문자열을 토크나이즈하여 토큰 ID로 변환
            target_token_ids = tokenizer(target, add_special_tokens=False)["input_ids"][1:]
            # print(f"\nTarget String: '{target}'")
            # print(f"Target Token IDs: {target_token_ids}")

            # 대상 문자열의 모든 시작 위치 찾기
            target_indices = []
            for i in range(len(batch) - len(target_token_ids) + 1):
                if batch[i:i+len(target_token_ids)].tolist() == target_token_ids:
                    target_indices.append(i)  # 시작 인덱스를 추가
            # print(f"Matched Indices for '{target}': {target_indices}")
            batch_result[target] = target_indices if target_indices else -1  # 없으면 -1 저장


        batch_indices.append(batch_result)  # 배치별 결과 저장
    # print(f"Batch Result: {batch_result}")


    return batch_indices


def mixed_fr_untargeted_gd_loss(model, ref_model, inputs):
    retain_inputs = inputs[15]  ##forget+retain_nomask_untargeted
    input_ids, labels, attention_mask = retain_inputs

    # tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    # decode_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    # print(f"Decode sentence_fr_unt: {decode_text}")

    # Main model outputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

    # Reference model outputs
    with torch.no_grad():
        outputs_ref = ref_model(input_ids, labels=labels, attention_mask=attention_mask)

    # Compute KL divergence between main and reference model
    loss = get_mixed_untargeted_loss3(
        labels, outputs.logits, input_ids, ref_logits=outputs_ref.logits
    )

    return loss

def mixed_rf_untargeted_gd_loss(model, ref_model, inputs):
    retain_inputs = inputs[16]  ##retain+forget_nomask_untargeted
    input_ids, labels, attention_mask = retain_inputs

    # tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    # decode_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    # print(f"Decode sentence_rf_unt: {decode_text}")

    # Main model outputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

    # Reference model outputs
    with torch.no_grad():
        outputs_ref = ref_model(input_ids, labels=labels, attention_mask=attention_mask)

    # Compute KL divergence between main and reference model
    loss = get_mixed_untargeted_loss4(
        labels, outputs.logits, input_ids, attention_mask, ref_logits=outputs_ref.logits
    )

    return loss

def get_mixed_untargeted_loss3(labels, logits, input_ids, ref_logits):
    """
    logits: 메인 모델의 출력 (bs, seq_len, vocab_size)
    input_ids: 입력 토큰 ID (bs, seq_len)
    ref_logits: 기준 모델의 출력 (bs, seq_len, vocab_size)
    """
    # 타겟 문자열들의 모든 토큰 인덱스 찾기
    indices = find_all_token_indices_batch1(input_ids)
    num_labels = logits.shape[-1]

    # Adjust logits and input_ids to exclude the last token
    input_ids = input_ids[:, 1:].clone()  # (bs, seq_len - 1)
    logits = logits[:, :-1, :]  # (bs, seq_len - 1, vocab_size)
    ref_logits = ref_logits[:, :-1, :]  # 기준 모델도 동일하게 조정

    # Flatten logits and input_ids for processing
    batch_size, seq_len_minus_one = input_ids.shape
    flat_logits = logits.reshape(-1, num_labels)
    flat_ref_logits = ref_logits.reshape(-1, num_labels)
    flat_input_ids = input_ids.reshape(-1)

    # 기준 모델에 유니폼 분포 강제
    uniform_dist = torch.full_like(flat_ref_logits, 1.0 / num_labels)

    # 원핫 벡터 생성 (input_ids 기반)
    one_hot_inputs = torch.zeros_like(flat_ref_logits)  # Same shape as flat_ref_logits
    one_hot_inputs.scatter_(1, flat_input_ids.unsqueeze(1), 1.0)  # 원핫 라벨링


    for batch_idx in range(batch_size):
        ranges = []
        batch_indices = indices[batch_idx]
        if " [/INST]" in batch_indices and "\n" in batch_indices:
            one_positions = batch_indices[" [/INST]"]
            newline_positions = batch_indices["\n"]
            start_positions = batch_indices["[INST] "]
            for i, pos in enumerate(start_positions):
                if i == 0:
                    ranges.append((pos, newline_positions[0]))
            for i, pos in enumerate(one_positions):
                if i == 0:  
                    ranges.append((pos, newline_positions[1]))
        # print(f"fr:{input_ids[batch_idx][70:170]}")
        # tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
        # decode_text = tokenizer.decode(input_ids[batch_idx], skip_special_tokens=True)
        # print(f"Decode sentence: {decode_text}")
        # Apply uniform distribution to specific ranges in ref_logits
        # print(f"ranges_fr:{ranges}")
        for i, (start, end) in enumerate(ranges):  # ranges는 [(50, 118)]
            if i==0:
                # print(f"Applying uniform distribution from {start} to {end}")
                for idx in range(start+5, end):  # 범위에 포함된 모든 토큰 처리
                    flat_idx = batch_idx * seq_len_minus_one + idx
                    if flat_idx < flat_ref_logits.size(0):  # 유효한 flat_idx인지 확인
                        flat_ref_logits[flat_idx] = uniform_dist[flat_idx]
                    else:
                        print(f"Warning: flat_idx {flat_idx} is out of range")
            elif i==1:
                # print(f"Applying uniform distribution from {start} to {end}")
                for idx in range(start+6, end):  # 범위에 포함된 모든 토큰 처리
                    flat_idx = batch_idx * seq_len_minus_one + idx
                    if flat_idx < flat_ref_logits.size(0):  # 유효한 flat_idx인지 확인
                        flat_ref_logits[flat_idx] = uniform_dist[flat_idx]
                    else:
                        print(f"Warning: flat_idx {flat_idx} is out of range")
    

    # 유니폼 분포가 적용되지 않은 부분에 원핫 라벨링 적용
    for flat_idx in range(flat_ref_logits.size(0)):
        if flat_ref_logits[flat_idx].tolist() != uniform_dist[flat_idx].tolist():
            flat_ref_logits[flat_idx] = one_hot_inputs[flat_idx]
# ############################################################################################################
    # # Compute KL divergence between main model and reference model
    # print("\n=== Debug:RF ===")
    # print(f"flat_logits shape: {flat_logits.shape}, flat_ref_logits shape: {flat_ref_logits.shape}")
    # # 배치 크기 제한 (첫 2개 배치만 확인)
    # batch_size = logits.size(0)
    # seq_len_minus_one = logits.size(1)
    # num_labels = logits.size(-1)

    # for batch_idx in range(min(batch_size, 2)):  # 첫 2개 배치만 출력
    #     print(f"\n[Batch {batch_idx}]")
    #     for seq_idx in range(50,min(seq_len_minus_one, 100)):  # 각 배치의 첫 3개 토큰
    #         flat_idx = batch_idx * seq_len_minus_one + seq_idx
    #         print(f"  Token {seq_idx}:")
    #         print(f"    flat_logits[:5]: {flat_logits[flat_idx][:5].tolist()}")
    #         print(f"    flat_ref_logits[:5]: {flat_ref_logits[flat_idx][:5].tolist()}")
# ############################################################################################################
    # Softmax로 확률 분포 계산
    soft_outputs = F.softmax(flat_logits, dim=-1)  # 확률 분포
    soft_ref_outputs = F.softmax(flat_ref_logits, dim=-1)  # 기준 확률 분포

    # Softmax 값 클리핑 (수치 안정성 확보)
    soft_outputs = torch.clamp(soft_outputs, min=1e-12, max=1.0)
    soft_ref_outputs = torch.clamp(soft_ref_outputs, min=1e-12, max=1.0)

    # KL-divergence 계산
    kl_div = (soft_outputs * (torch.log(soft_outputs) - torch.log(soft_ref_outputs))).sum(-1)

    labels = labels[:, 1:].clone()
    loss_mask = (labels != -100).view(-1)
    masked_kl_div = kl_div *loss_mask

    # 전체 평균 KL-divergence
    loss = masked_kl_div.sum() / loss_mask.sum()
    # print(f"kl_div:{kl_div}")
    # print(f"masked_kl_div:{masked_kl_div}")
    # print(f"loss: {loss}")

    return loss
def get_mixed_untargeted_loss4(labels, logits, input_ids, attention_mask, ref_logits):
    """
    logits: 메인 모델의 출력 (bs, seq_len, vocab_size)
    input_ids: 입력 토큰 ID (bs, seq_len)
    ref_logits: 기준 모델의 출력 (bs, seq_len, vocab_size)
    """
    # 타겟 문자열들의 모든 토큰 인덱스 찾기
    indices = find_all_token_indices_batch2(input_ids, attention_mask)
    num_labels = logits.shape[-1]

    # Adjust logits and input_ids to exclude the last token
    input_ids = input_ids[:, 1:].clone()  # (bs, seq_len - 1)
    logits = logits[:, :-1, :]  # (bs, seq_len - 1, vocab_size)
    ref_logits = ref_logits[:, :-1, :]  # 기준 모델도 동일하게 조정

    # Flatten logits and input_ids for processing
    batch_size, seq_len_minus_one = input_ids.shape
    flat_logits = logits.reshape(-1, num_labels)
    flat_input_ids = input_ids.reshape(-1)
    flat_ref_logits = torch.zeros_like(flat_logits)

    # 원핫 벡터 생성 (input_ids 기반)
    flat_ref_logits.scatter_(1, flat_input_ids.unsqueeze(1), 1.0)

    # 유니폼 분포 생성
    uniform_dist = torch.full_like(flat_logits, 1.0 / num_labels)

    # 특정 범위에 유니폼 분포 적용
    for batch_idx in range(batch_size):
        ranges = []
        batch_indices = indices[batch_idx]
        # print(batch_indices)
        # if "\n" in batch_indices:
        #     sub_dict = batch_indices["\n"]

        #     if "second_2_start" in sub_dict and "valid_end" in sub_dict:
        #         one_position = sub_dict["second_2_start"]
        #         newline_position = sub_dict["valid_end"]

        #         ranges.append((one_position, newline_position))
        # print(f"batch_indices: {batch_indices}")
        one_positions = batch_indices["\n"]
        newline_positions1 = batch_indices[" [/INST]"]
        newline_positions2 = batch_indices["valid_end"]
        for i, pos in enumerate(one_positions):
            if i ==0:
                ranges.append((pos,newline_positions1[0]))
            elif i==1:
                ranges.append((pos,newline_positions2))
        # print(f"rf:{input_ids[batch_idx][70:170]}")
        # tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
        # decode_text = tokenizer.decode(input_ids[batch_idx], skip_special_tokens=True)
        # print(f"Decode sentence: {decode_text}")
        # print(f"ranges_rf: {ranges}")
        # Apply uniform distribution to specific ranges in flat_ref_logits
        for start, end in ranges:
            for idx in range(start + 3, end):
                flat_idx = batch_idx * seq_len_minus_one + idx
                if flat_idx < flat_ref_logits.size(0):  # 범위를 초과하지 않는 경우에만 유니폼 적용
                    flat_ref_logits[flat_idx] = uniform_dist[flat_idx]

    # Softmax로 확률 분포 계산
    soft_outputs = F.softmax(flat_logits, dim=-1)  # 확률 분포
    soft_ref_outputs = F.softmax(flat_ref_logits, dim=-1)  # 기준 확률 분포

    # Softmax 값 클리핑 (수치 안정성 확보)
    soft_outputs = torch.clamp(soft_outputs, min=1e-12, max=1.0)
    soft_ref_outputs = torch.clamp(soft_ref_outputs, min=1e-12, max=1.0)
    
    # KL-divergence 계산
    kl_div = (soft_outputs * (torch.log(soft_outputs) - torch.log(soft_ref_outputs))).sum(-1)

    labels = labels[:, 1:].clone()
    loss_mask = (labels != -100).view(-1)
    masked_kl_div = kl_div *loss_mask

    # 전체 평균 KL-divergence
    loss = masked_kl_div.sum() / loss_mask.sum()
    # print(f"kl_div:{kl_div}")
    # print(f"masked_kl_div:{masked_kl_div[20:70]}")
    # print(f"loss: {loss}")

    return loss

def get_mixed_untargeted_loss5(labels, logits, input_ids, ref_logits):
    """
    logits: 메인 모델의 출력 (bs, seq_len, vocab_size)
    input_ids: 입력 토큰 ID (bs, seq_len)
    ref_logits: 기준 모델의 출력 (bs, seq_len, vocab_size)
    """
    # 1) 타겟 문자열들의 모든 토큰 인덱스 찾기
    indices = find_all_token_indices_batch1(input_ids)
    num_labels = logits.shape[-1]

    # 2) 마지막 토큰 제외
    input_ids = input_ids[:, 1:].clone()      # (bs, seq_len - 1)
    logits    = logits[:, :-1, :]            # (bs, seq_len - 1, vocab_size)
    ref_logits= ref_logits[:, :-1, :]        # 기준 모델도 동일하게 조정

    # 3) Flatten
    batch_size, seq_len_minus_one = input_ids.shape
    flat_logits     = logits.reshape(-1, num_labels)      # (bs * (seq_len-1), vocab_size)
    flat_ref_logits = ref_logits.reshape(-1, num_labels)  # (bs * (seq_len-1), vocab_size)
    flat_input_ids  = input_ids.reshape(-1)               # (bs * (seq_len-1))

    # 4) one-hot 벡터 생성
    one_hot_inputs = torch.zeros_like(flat_ref_logits)  # 기준크기와 동일
    one_hot_inputs.scatter_(1, flat_input_ids.unsqueeze(1), 1.0)

    # 5) 특정 범위 내 -> one-hot 로짓
    #    범위 밖 -> ref_logits 원본 유지
    for batch_idx in range(batch_size):
        ranges = []
        batch_indices = indices[batch_idx]

        # 예시) [INST]~, [/INST], \n 등 특정 토큰들의 위치를 찾아 ranges 생성
        if " [/INST]" in batch_indices and "\n" in batch_indices:
            one_positions     = batch_indices[" [/INST]"]
            newline_positions = batch_indices["\n"]
            start_positions   = batch_indices["[INST] "]
            # 아래는 예시 로직 - 실제 로직은 사용자 정의
            for i, pos in enumerate(start_positions):
                if i == 0:
                    ranges.append((pos, newline_positions[0]))
            for i, pos in enumerate(one_positions):
                if i == 0:
                    ranges.append((pos, newline_positions[1]))

        # print(f"ranges_fr:{ranges}")
        for i, (start, end) in enumerate(ranges):
            # i==0, i==1 등에 따라 다른 오프셋 적용
            offset = 5 if i == 0 else 6
            for idx in range(start + offset, end):
                flat_idx = batch_idx * seq_len_minus_one + idx
                if 0 <= flat_idx < flat_ref_logits.size(0):
                    # "범위 내"는 one-hot 로짓으로 교체
                    flat_ref_logits[flat_idx] = one_hot_inputs[flat_idx]
# ############################################################################################################
    # # # Compute KL divergence between main model and reference model
    # print("\n=== Debug:FR ===")
    # print(f"flat_logits shape: {flat_logits.shape}, flat_ref_logits shape: {flat_ref_logits.shape}")
    # # 배치 크기 제한 (첫 2개 배치만 확인)
    # batch_size = logits.size(0)
    # seq_len_minus_one = logits.size(1)
    # num_labels = logits.size(-1)

    # for batch_idx in range(min(batch_size, 2)):  # 첫 2개 배치만 출력
    #     print(f"\n[Batch {batch_idx}]")
    #     for seq_idx in range(0,min(seq_len_minus_one, 50)):  # 각 배치의 첫 3개 토큰
    #         flat_idx = batch_idx * seq_len_minus_one + seq_idx
    #         print(f"  Token {seq_idx}:")
    #         print(f"    flat_logits[:5]: {flat_logits[flat_idx][:5].tolist()}")
    #         print(f"    flat_ref_logits[:5]: {flat_ref_logits[flat_idx][:5].tolist()}")
# ############################################################################################################ 
    # 6) KL Divergence 계산
    #    - 범위 밖은 ref_logits 원본,
    #    - 범위 내는 one-hot 된 상태
    soft_outputs     = F.softmax(flat_logits, dim=-1)       # p
    soft_ref_outputs = F.softmax(flat_ref_logits, dim=-1)   # q
    soft_outputs     = torch.clamp(soft_outputs, min=1e-12, max=1.0)
    soft_ref_outputs = torch.clamp(soft_ref_outputs, min=1e-12, max=1.0)

    kl_div = (soft_outputs * (torch.log(soft_outputs) - torch.log(soft_ref_outputs))).sum(-1)

    # 7) 마스크 적용: label=-100 부분 무시
    labels = labels[:, 1:].clone()
    loss_mask = (labels != -100).view(-1)
    masked_kl_div = kl_div * loss_mask

    loss = masked_kl_div.sum() / loss_mask.sum()
    return loss

def get_mixed_untargeted_loss6(labels, logits, input_ids, attention_mask, ref_logits):
    """
    logits: 메인 모델의 출력 (bs, seq_len, vocab_size)
    input_ids: 입력 토큰 ID (bs, seq_len)
    ref_logits: 기준 모델의 출력 (bs, seq_len, vocab_size)
    """
    # 1) indices: 특정 토큰들의 인덱스 (사용자 정의)
    indices = find_all_token_indices_batch2(input_ids, attention_mask)
    num_labels = logits.shape[-1]

    # 2) 마지막 토큰 제외
    input_ids  = input_ids[:, 1:].clone()    # (bs, seq_len - 1)
    logits     = logits[:, :-1, :]          # (bs, seq_len - 1, vocab_size)
    ref_logits = ref_logits[:, :-1, :]      # (bs, seq_len - 1, vocab_size)

    # 3) Flatten
    batch_size, seq_len_minus_one = input_ids.shape
    flat_logits     = logits.reshape(-1, num_labels)
    flat_input_ids  = input_ids.reshape(-1)
    flat_ref_logits = ref_logits.reshape(-1, num_labels)   # 원본 ref_logits 그대로 사용

    # 4) One-hot 벡터 (input_ids 기반)
    one_hot_inputs = torch.zeros_like(flat_ref_logits)
    one_hot_inputs.scatter_(1, flat_input_ids.unsqueeze(1), 1.0)

    # 5) 주어진 ranges 범위만 "one-hot" 로 교체
    #    그 외 범위는 원본 ref_logits
    for batch_idx in range(batch_size):
        batch_indices = indices[batch_idx]
        ranges = []

        # 예: batch_indices 구조 예시
        # {
        #    "\n": [10, 32],
        #    " [/INST]": [50],
        #    "valid_end": 80
        #    ...
        # }
        one_positions        = batch_indices["\n"]          # 예시
        newline_positions_1  = batch_indices[" [/INST]"]
        newline_positions_2  = batch_indices["valid_end"]

        # 예시 로직
        # 첫 번째 '\n' ~ 첫 번째 ' [/INST]'
        # 두 번째 '\n' ~ 'valid_end'
        for i, pos in enumerate(one_positions):
            if i == 0:
                ranges.append((pos, newline_positions_1[0]))
            elif i == 1:
                ranges.append((pos, newline_positions_2))

        # print(f"ranges_rf: {ranges}")
        for start, end in ranges:
            # 범위 내 -> one-hot 치환
            for idx in range(start + 3, end):
                flat_idx = batch_idx * seq_len_minus_one + idx
                if 0 <= flat_idx < flat_ref_logits.size(0):
                    flat_ref_logits[flat_idx] = one_hot_inputs[flat_idx]
                else:
                    print(f"Warning: flat_idx {flat_idx} is out of range")
# ############################################################################################################
    # # # # Compute KL divergence between main model and reference model
    # print("\n=== Debug:RF ===")
    # print(f"flat_logits shape: {flat_logits.shape}, flat_ref_logits shape: {flat_ref_logits.shape}")
    # # 배치 크기 제한 (첫 2개 배치만 확인)
    # batch_size = logits.size(0)
    # seq_len_minus_one = logits.size(1)
    # num_labels = logits.size(-1)

    # for batch_idx in range(min(batch_size, 2)):  # 첫 2개 배치만 출력
    #     print(f"\n[Batch {batch_idx}]")
    #     for seq_idx in range(0,min(seq_len_minus_one, 130)):  # 각 배치의 첫 3개 토큰
    #         flat_idx = batch_idx * seq_len_minus_one + seq_idx
    #         print(f"  Token {seq_idx}:")
    #         print(f"    flat_logits[:5]: {flat_logits[flat_idx][:5].tolist()}")
    #         print(f"    flat_ref_logits[:5]: {flat_ref_logits[flat_idx][:5].tolist()}")
# ############################################################################################################    
    # 6) KL-divergence 계산
    soft_outputs     = F.softmax(flat_logits,     dim=-1)
    soft_ref_outputs = F.softmax(flat_ref_logits, dim=-1)
    soft_outputs     = torch.clamp(soft_outputs,     min=1e-12, max=1.0)
    soft_ref_outputs = torch.clamp(soft_ref_outputs, min=1e-12, max=1.0)

    kl_div = (soft_outputs * (torch.log(soft_outputs) - torch.log(soft_ref_outputs))).sum(-1)

    # 7) 라벨 -100 마스킹
    labels     = labels[:, 1:].clone()
    loss_mask  = (labels != -100).view(-1)
    masked_kl_div = kl_div * loss_mask

    loss = masked_kl_div.sum() / loss_mask.sum()
    return loss


def mixed_fr_targeted_kl_loss(model, ref_model, inputs):
    retain_inputs = inputs[17]  ##forget+retain_nomask
    input_ids, labels, attention_mask = retain_inputs

    # tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    # decode_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    # print(f"Decode sentence_fr_unt: {decode_text}")

    # Main model outputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

    # Reference model outputs
    with torch.no_grad():
        outputs_ref = ref_model(input_ids, labels=labels, attention_mask=attention_mask)

    # Compute KL divergence between main and reference model
    loss = get_mixed_untargeted_loss5(
        labels, outputs.logits, input_ids, ref_logits=outputs_ref.logits
    )

    return loss

def mixed_rf_targeted_kl_loss(model, ref_model, inputs):
    retain_inputs = inputs[18]  ##retain+forget_nomask
    input_ids, labels, attention_mask = retain_inputs

    # tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    # decode_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    # print(f"Decode sentence_rf_unt: {decode_text}")

    # Main model outputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

    # Reference model outputs
    with torch.no_grad():
        outputs_ref = ref_model(input_ids, labels=labels, attention_mask=attention_mask)

    # Compute KL divergence between main and reference model
    loss = get_mixed_untargeted_loss6(
        labels, outputs.logits, input_ids, attention_mask, ref_logits=outputs_ref.logits
    )

    return loss

def fr_ugad_loss(model, inputs):
    input1 = inputs[19]
    input_ids1, labels1, attention_mask1 = input1
    output1 = model(input_ids1, labels=labels1, attention_mask=attention_mask1)
    input2 = inputs[20]
    input_ids2, labels2, attention_mask2 = input2
    output2 = model(input_ids2, labels=labels2, attention_mask=attention_mask2)
    loss = -1 * output1.loss + output2.loss
    return loss

def rf_ugad_loss(model, inputs):
    input1 = inputs[21]
    input_ids1, labels1, attention_mask1 = input1
    output1 = model(input_ids1, labels=labels1, attention_mask=attention_mask1)
    input2 = inputs[22]
    input_ids2, labels2, attention_mask2 = input2
    output2 = model(input_ids2, labels=labels2, attention_mask=attention_mask2)
    loss = -1 * output1.loss + output2.loss
    return loss
