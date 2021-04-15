import torch
import torch.nn as nn

from model_utils import getPrincipalComponents, removeComponent

class DetoxRobertaClassifier(nn.Module):

    def __init__(self, cls_model, bert, gpu):
        super(DetoxRobertaClassifier, self).__init__()
        self.roberta = cls_model.roberta
        self.bert = bert
        self.emb = self.roberta.embeddings
        self.blocks = self.roberta.encoder.layer

        self.fc = cls_model.classifier

    def forward(self, tox, ntox, attn_masks):

        ev = [0] * 13
        u = self.emb(tox, attn_masks)
        v = self.emb(ntox, attn_masks)
        D = u - v  # todo: check dim, torch. ?
        pc, ev[0] = getPrincipalComponents(D)
        pc = pc.cuda()

        for j in range(1, 13):
            if self.debias:
                uproj = removeComponent(u[0], pc)
                vproj = removeComponent(v[0], pc)
            else:
                uproj = u[0]
                vproj = v[0]

            uproj = uproj.unsqueeze(dim=0).cuda()
            vproj = vproj.unsqueeze(dim=0).cuda()

            device = uproj.device
            input_shape = uproj.size()
            ext_attn_mask: torch.Tensor = self.bert.get_extended_attention_mask(attn_masks, input_shape, device)
            u = self.layers[j - 1](uproj, ext_attn_mask)[0]
            v = self.layers[j - 1](vproj, ext_attn_mask)[0]
            D = u - v  # todo: check dim, torch. ?
            pc, ev[j] = getPrincipalComponents(D)
            pc = pc.cuda()

        out = self.fc(u)
        return out