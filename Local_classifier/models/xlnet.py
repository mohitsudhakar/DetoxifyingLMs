import torch
import torch.nn as nn


class DetoxXlnetClassifier(nn.Module):

    def __init__(self, cls_model):
        super(DetoxXlnetClassifier, self).__init__()
        self.xln = cls_model.transformer
        self.emb = self.xln.word_embedding
        self.blocks = self.xln.layer
        self.dropout = self.xln.dropout
        self.dtype = self.xln.dtype

        self.fc = nn.Sequential(cls_model.sequence_summery, cls_model.logits_proj)

    def forward(self, tox, ntox, attn_mask):

        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # so we move here the first dimension (batch) to the end

        tox = tox.transpose(0, 1).contiguous()
        ntox = ntox.transpose(0, 1).contiguous()
        qlen, bsz = tox.shape[0], tox.shape[1]

        attn_mask = attn_mask.transpose(0, 1).contiguous()
        mlen = 0
        klen = mlen + qlen
        dtype_float = self.dtype
        device = self.device

        # data mask: input mask & perm mask
        input_mask = 1.0 - attn_mask
        data_mask = input_mask[None]

        if data_mask is not None:
            attn_mask += data_mask[:, :, :, None]

        attn_mask = (attn_mask > 0).to(dtype_float)
        non_tgt_mask = -torch.eye(qlen).to(attn_mask)
        non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask)

        t_word_emb_k = self.word_embedding(tox)
        nt_word_emb_k = self.word_embedding(ntox)
        t_output_h = self.dropout(t_word_emb_k)
        nt_output_h = self.dropout(nt_word_emb_k)
        t_output_g, nt_output_g = None, None
        seg_mat = None

        # Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb)

        for i, layer_module in enumerate(self.layer):
            t_outputs = layer_module(t_output_h, t_output_g, attn_mask_h=non_tgt_mask,
                                     attn_mask_g=attn_mask, r=pos_emb, seg_mat=seg_mat)
            t_output_h, t_output_g = t_outputs[:2]
            nt_outputs = layer_module(nt_output_h, nt_output_g, attn_mask_h=non_tgt_mask,
                                      attn_mask_g=attn_mask, r=pos_emb, seg_mat=seg_mat)
            nt_output_h, nt_output_g = nt_outputs[:2]

        t_output = self.dropout(t_output_g if t_output_g is not None else t_output_h)
        nt_output = self.dropout(nt_output_g if nt_output_g is not None else nt_output_h)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        t_output = t_output.permute(1, 0, 2).contiguous()
        nt_output = nt_output.permute(1, 0, 2).contiguous()

        output = self.sequence_summary(t_output)
        output = self.logits_proj(output)
        return output