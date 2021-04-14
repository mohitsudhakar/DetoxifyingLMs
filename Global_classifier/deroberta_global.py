import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import model_utils


class RobertaGlobalClassifier(nn.Module):
    def __init__(self, freeze_weights=False):
        super(RobertaGlobalClassifier, self).__init__()
        _, self.roberta, _ = model_utils.initRoberta(freeze_weights)
        self.fc = nn.Linear(768, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attn_mask):
        out = self.roberta(input_ids, attn_mask)
        out = out.pooler_output
        out = self.fc(out)
        return out


class DeRobertaGlobalClassifier(nn.Module):

    def __init__(self, bias_subspace, freeze_weights=False):
        super(DeRobertaGlobalClassifier, self).__init__()
        _, self.roberta, _ = model_utils.initRoberta(freeze_weights)
        self.subspace = bias_subspace
        self.fc = nn.Linear(768, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attn_mask):
        out = self.roberta(input_ids, attn_mask)
        out = out.pooler_output
        out = model_utils.projection(out, self.subspace)
        out = self.fc(out)
        return out


if __name__ == '__main__':

    """
    Test case with one sentence

    1. Put sentence through tokenizer and model
    2. Call projection() on model output (a) and pc_tensor (b)
    3. Output is debiased rep, can be passed downstream
    """
    tokenizer, _, _ = model_utils.initRoberta()
    device = 'cpu'
    data_path = '../data/'
    pc_file = data_path + 'princComp.txt'

    loaded_list = np.loadtxt(pc_file, delimiter=" ")
    pcs = torch.FloatTensor(loaded_list)

    sentence = ["What the fuck is wrong with you? Is that even possible you fuck!"]
    inputs = tokenizer(
        sentence, add_special_tokens=True, truncation=True,
        max_length=128, padding='max_length',
        return_attention_mask=True,
        return_tensors='pt')

    cls1 = RobertaGlobalClassifier()
    cls1 = cls1.to(device)

    cls2 = DeRobertaGlobalClassifier(pcs)
    cls2 = cls2.to(device)

    out1 = cls1(inputs['input_ids'], inputs['attention_mask'])
    out2 = cls2(inputs['input_ids'], inputs['attention_mask'])

    labels = torch.tensor(np.array([1]))
    criterion = nn.CrossEntropyLoss()
    loss1 = criterion(out1, labels)
    loss2 = criterion(out2, labels)

    print('loss normal', loss1, 'loss debiased', loss2)
    print('pred normal', out1.argmax(1))
    print('pred deb', out2.argmax(1))

    print('Normal cls output', out1.shape, out1)
    print('Debiased cls output', out2.shape, out2)

