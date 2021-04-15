import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import model_utils

class GPT2GlobalClassifier(nn.Module):
    def __init__(self, freeze_weights=False):
        super(GPT2GlobalClassifier, self).__init__()
        self.tokenizer, self.model = model_utils.initGpt2(freeze_weights)
        self.fc = nn.Linear(768, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attn_mask):
        out = self.model(input_ids=input_ids, attention_mask=attn_mask)
        out = out.last_hidden_state
        out = self.fc(out)
        batch_size = input_ids.shape[0]
        sequence_lengths = torch.ne(input_ids, self.tokenizer.pad_token_id).sum(-1) - 1
        pooled_out = out[range(batch_size), sequence_lengths]

        return pooled_out

#
# class GPT2GlobalClassifier(nn.Module):
#     def __init__(self, freeze_weights=False):
#         super(GPT2GlobalClassifier, self).__init__()
#         self.tokenizer, self.model = model_utils.initGpt2(freeze_weights)
#         self.fc = nn.Linear(768, 2)
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, input_ids, attn_mask):
#         out = self.model(input_ids=input_ids, attention_mask=attn_mask)
#         out = out.last_hidden_state
#         # print(out.shape)
#         batch_size = input_ids.shape[0]
#         sequence_lengths = torch.ne(input_ids, self.tokenizer.pad_token_id).sum(-1) - 1
#         # print('seq lens', sequence_lengths)
#         pooled_out = out[range(batch_size), sequence_lengths]
#         # print(pooled_out.shape)
#         out = self.fc(pooled_out)
#         # print(out.shape)
#
#         return out


class DeGPT2GlobalClassifier(nn.Module):

    def __init__(self, bias_subspace, freeze_weights=False):
        super(DeGPT2GlobalClassifier, self).__init__()
        self.tokenizer, self.model = model_utils.initGpt2(freeze_weights)
        self.subspace = bias_subspace
        self.fc = nn.Linear(768, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attn_mask):
        out = self.model(input_ids=input_ids, attention_mask=attn_mask)
        out = out.last_hidden_state

        batch_size = input_ids.shape[0]
        sequence_lengths = torch.ne(input_ids, self.tokenizer.pad_token_id).sum(-1) - 1
        pooled_out = out[range(batch_size), sequence_lengths]

        out = model_utils.removeComponent(pooled_out, self.subspace)
        out = self.fc(out)
        return out


if __name__ == '__main__':

    """
    Test case with one sentence

    1. Put sentence through tokenizer and model
    2. Call projection() on model output (a) and pc_tensor (b)
    3. Output is debiased rep, can be passed downstream
    """
    tokenizer, model = model_utils.initGpt2()
    device = 'cpu'
    data_path = '../data/'
    pc_file = data_path + 'princComp.txt'

    loaded_list = np.loadtxt(pc_file, delimiter=" ")
    pcs = torch.FloatTensor(loaded_list)

    # sentence = ["What the fuck is wrong with you? Is that even possible you fuck!"]
    sentence = ['Now, a new documentary by Danish broadcaster TV2 has reignited debate over whether the infamous mosque should be shut down.']
    # inputs = tokenizer(sentence, return_tensors='pt')
    inputs = tokenizer(
        sentence, add_special_tokens=True, truncation=True,
        max_length=128, padding='max_length',
        return_attention_mask=True,
        return_tensors='pt')
    print('Inputs',inputs)

    cls1 = GPT2GlobalClassifier(True)
    cls1 = cls1.to(device)

    cls2 = DeGPT2GlobalClassifier(pcs)
    cls2 = cls2.to(device)

    out1 = cls1(inputs['input_ids'], inputs['attention_mask'])
    out2 = cls2(inputs['input_ids'], inputs['attention_mask'])
    print("Outputs::::")
    print(out1, out2)
    labels = torch.tensor(np.array([1]))
    criterion = nn.CrossEntropyLoss()
    loss1 = criterion(out1, labels)
    loss2 = criterion(out2, labels)

    print('loss normal', loss1)
    print('loss debiased', loss2)
    print('pred normal', out1.argmax(1))
    print('pred deb', out2.argmax(1))

    print('Normal cls output', out1.shape, out1)
    print('Debiased cls output', out2.shape, out2)

