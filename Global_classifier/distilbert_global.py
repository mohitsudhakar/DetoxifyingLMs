import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import model_utils


class DistilBertGlobalClassifier(nn.Module):
    def __init__(self, freeze_weights=False):
        super(DistilBertGlobalClassifier, self).__init__()
        _, self.model = model_utils.initDistilBert(freeze_weights)
        self.pre_classifier = nn.Linear(768, 768)
        self.fc = nn.Linear(768, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attn_mask):
        out = self.model(input_ids, attn_mask)

        hidden_state = out.last_hidden_state  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.fc(pooled_output)  # (bs, num_labels)

        return logits


class DeDistilBertGlobalClassifier(nn.Module):

    def __init__(self, bias_subspace, freeze_weights=False):
        super(DeDistilBertGlobalClassifier, self).__init__()
        _, self.model = model_utils.initDistilBert(freeze_weights)
        self.subspace = bias_subspace
        self.pre_classifier = nn.Linear(768, 768)
        self.fc = nn.Linear(768, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attn_mask):

        out = self.model(input_ids, attn_mask)
        hidden_state = out.last_hidden_state  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        # remove first
        out = model_utils.removeComponent(pooled_output, self.subspace)

        out = self.pre_classifier(out)  # (bs, dim)
        out = nn.ReLU()(out)  # (bs, dim)
        out = self.dropout(out)  # (bs, dim)
        logits = self.fc(out)  # (bs, num_labels)

        return logits



if __name__ == '__main__':

    """
    Test case with one sentence

    1. Put sentence through tokenizer and model
    2. Call projection() on model output (a) and pc_tensor (b)
    3. Output is debiased rep, can be passed downstream
    """
    tokenizer, model = model_utils.initDistilBert()
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

    cls1 = DistilBertGlobalClassifier()
    cls1 = cls1.to(device)

    cls2 = DeDistilBertGlobalClassifier(pcs)
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

