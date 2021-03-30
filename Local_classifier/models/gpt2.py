import torch
import torch.nn as nn
from transformers import GPT2Model

from model_utils import getPrincipalComponents, projection, initGpt2


class Gpt2Classifier(nn.Module):
    def __init__(self):
        super(Gpt2Classifier, self).__init__()
        _, self.model = initGpt2()
        self.fc = nn.Linear(768, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inp, attn_masks):
        out = self.model(inp, attn_masks)
        out = out.pooler_output
        out = self.fc(out)
        return out


class DeGpt2Classifier(nn.Module):

    def __init__(self, debias=False):
        super(DeGpt2Classifier, self).__init__()
        _, self.gpt2 = initGpt2()
        self.wte = self.gpt2.wte
        self.wpe = self.gpt2.wpe
        self.drop = self.gpt2.drop

        self.blocks = self.gpt2.h
        self.dtype = self.gpt2.dtype

        self.ln_f = self.gpt2.ln_f
        self.fc = nn.Linear(768, 2)
        self.debias = debias


    def forward(self, tox, ntox, attn_masks):
        input_shape = tox.size()
        tox = tox.view(-1, input_shape[-1])
        ntox = ntox.view(-1, input_shape[-1])
        device = tox.device
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        t_inputs_embeds = self.wte(tox)
        nt_inputs_embeds = self.wte(ntox)
        position_embeds = self.wpe(position_ids)

        t_hidden_states = t_inputs_embeds + position_embeds
        nt_hidden_states = nt_inputs_embeds + position_embeds

        t_hidden_states = self.drop(t_hidden_states)
        nt_hidden_states = self.drop(nt_hidden_states)

        ev = [0] * 13
        u = t_hidden_states
        v = nt_hidden_states
        D = u - v  # todo: check dim, torch. ?
        pc, ev[0] = getPrincipalComponents(D)
        pc = pc.cuda()

        for j in range(1, 13):
            if self.debias:
                uproj = projection(u[0], pc)
                vproj = projection(v[0], pc)
            else:
                uproj = u[0]
                vproj = v[0]

            uproj = uproj.unsqueeze(dim=0).cuda()
            vproj = vproj.unsqueeze(dim=0).cuda()

            block = self.blocks[j - 1]
            u = block(uproj, attention_mask=attn_masks)
            v = block(vproj, attention_mask=attn_masks)
            D = u - v  # todo: check dim, torch. ?
            pc, ev[j] = getPrincipalComponents(D)
            pc = pc.cuda()

        out = self.ln_f(u)
        out = self.fc(out)
        return out
