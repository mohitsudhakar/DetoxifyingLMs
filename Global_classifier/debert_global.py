import torch
import torch.nn as nn
import torch.nn.functional as F

import model_utils


class BertGlobalClassifier(nn.Module):
    def __init__(self, device):
        super(BertGlobalClassifier, self).__init__()
        _, self.bert = model_utils.initBert()
        self.device = device
        self.fc = nn.Linear(768, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attn_mask):
        # input_ids = input_ids.to(self.device)
        # attn_mask = attn_mask.to(self.device)
        out = self.bert(input_ids, attn_mask)
        out = out.pooler_output
        out = self.fc(out)
        return out


class DeBertGlobalClassifier(nn.Module):

    def __init__(self, bias_subspace, device):
        super(DeBertGlobalClassifier, self).__init__()
        _, self.bert = model_utils.initBert()
        self.subspace = bias_subspace
        self.device = device
        self.fc = nn.Linear(768, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attn_mask):
        # input_ids = input_ids.to(self.device)
        # attn_mask = attn_mask.to(self.device)
        out = self.bert(input_ids, attn_mask)
        out = out.pooler_output
        out = model_utils.projection(out, self.subspace)
        out = self.fc(out)
        return out

