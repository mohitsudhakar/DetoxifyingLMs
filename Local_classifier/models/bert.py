import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import model_utils


class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        _, self.model = model_utils.initBert()
        self.fc = nn.Linear(768, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inp, attn_masks):

        out = self.model(inp, attn_masks)
        out = out.pooler_output
        out = self.fc(out)
        return out


class DeBertClassifier(nn.Module):
    def __init__(self, debias=True):
        super(DeBertClassifier, self).__init__()

        _, self.model = model_utils.initBert()
        model = self.model
        self.emb = model.embeddings
        self.layers = model.encoder.layer
        self.pooler = model.pooler
        self.fc = nn.Linear(768, 2)
        self.dropout = nn.Dropout(0.1)
        self.debias = debias

    def forward(self, inp_t, attn_masks, inp_nt):

        u = self.emb(inp_t, attn_masks)
        v = self.emb(inp_nt, attn_masks)
        D = u - v  # todo: check dim, torch. ?
        print('D.shape', D.shape)
        # Local - per sentence
        D = D.squeeze(dim=0)
        pc, _, _ = model_utils.getPrincipalComponents(D)
        # pc = pc.cuda()

        for j in range(1, 13):
          uproj, vproj = u, v
          if self.debias:
              uproj = model_utils.projection(u[0], pc)
              vproj = model_utils.projection(v[0], pc)
              uproj = uproj.unsqueeze(dim=0)
              vproj = vproj.unsqueeze(dim=0)

          # uproj = uproj.cuda()
          # vproj = vproj.cuda()

          device = uproj.device
          input_shape = uproj.size()
          ext_attn_mask: torch.Tensor = self.model.get_extended_attention_mask(attn_masks, input_shape, device)
          u = self.layers[j - 1](uproj, ext_attn_mask)[0]
          v = self.layers[j - 1](vproj, ext_attn_mask)[0]
          D = u - v  # todo: check dim, torch. ?
          # Local - per sentence
          D = D.squeeze(dim=0)
          pc, _, _ = model_utils.getPrincipalComponents(D)
          # pc = pc.cuda()

        out = self.pooler(u)
        # out = self.dropout(out)
        out = self.fc(out)
        print(out)
        return out



if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    pad = tokenizer.pad_token
    toxic = 'he was a stupid boy'
    ntoxic = toxic.replace('stupid', pad)
    toks = tokenizer([toxic], add_special_tokens=True, truncation=True,
                            max_length=128, padding='max_length', return_attention_mask=True)
    ntoks = tokenizer([ntoxic], add_special_tokens=True, truncation=True, max_length=128,
                      padding='max_length', return_attention_mask=True)

    toxic_ids = toks['input_ids']
    toxic_ids = torch.LongTensor(toxic_ids)

    ntoxic_ids = ntoks['input_ids']
    ntoxic_ids = torch.LongTensor(ntoxic_ids)

    attn = toks['attention_mask']
    attn = torch.LongTensor(attn)

    print(toxic_ids.shape, ntoxic_ids.shape, attn.shape)

    criterion = nn.CrossEntropyLoss()
    labels = torch.tensor([1])

    bert_cls = BertClassifier()
    output = bert_cls(toxic_ids, attn)
    print('bert cls', output)
    res, loss = output.argmax(1), criterion(output, labels)
    print('bert cls result', res, 'loss', loss)

    debert_cls = DeBertClassifier()
    output = debert_cls(toxic_ids, attn, ntoxic_ids)
    print('debert cls', output)
    res, loss = output.argmax(1), criterion(output, labels)
    print('debert cls result', res, 'loss', loss)
