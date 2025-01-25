import torch
import torch.nn as nn
import torch.nn.functional as F
import re

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