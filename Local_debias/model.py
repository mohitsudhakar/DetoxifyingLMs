import torch
import torch.nn as nn

class LayerBert(nn.Module):

    def __init__(self, base_model, gpu):
        super(LayerBert, self).__init__()
        self.bert = base_model.cuda(gpu)
        self.emb = self.bert.embeddings
        self.layers = self.bert.encoder.layer

    def forward(self, layer_num, inp_ids, attn_masks):
        if layer_num == 0:
            out = self.emb(inp_ids, attn_masks)  # both are batch_size x seq_len

        else:
            device = inp_ids.device
            input_shape = inp_ids.size()
            ext_attn_mask: torch.Tensor = self.bert.get_extended_attention_mask(attn_masks, input_shape, device)
            out = self.layers[layer_num - 1](inp_ids, ext_attn_mask)
            out = out[0]
            del ext_attn_mask

        torch.cuda.empty_cache()
        return out


class LayerGpt2(nn.Module):

    def __init__(self, base_model, gpu):
        super(LayerGpt2, self).__init__()
        self.gpt2 = base_model.cuda(gpu)
        self.wte = self.gpt2.wte
        self.wpe = self.gpt2.wpe
        self.drop = self.gpt2.drop

        self.blocks = self.gpt2.h

        self.dtype = self.gpt2.dtype

    def forward(self, layer_num, inp, attention_mask):

        if layer_num == 0:
            input_ids = inp
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            device = input_ids.device
            # print('device', device)
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

            # print('input_ids' , input_ids.shape)
            inputs_embeds = self.wte(input_ids)
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
            hidden_states = self.drop(hidden_states)

            del inputs_embeds, position_embeds
            torch.cuda.empty_cache()

            return hidden_states

        else:
            block = self.blocks[layer_num - 1]
            outputs = block(inp, attention_mask=attention_mask)
            hidden_states = outputs[0]
            return hidden_states

        # hidden_states = self.ln_f(hidden_states)
        # output_shape = input_shape + (hidden_states.size(-1),)
        # hidden_states = hidden_states.view(*output_shape)


class LayerRoberta(nn.Module):

    def __init__(self, base_model, bert_model, gpu):
        super(LayerRoberta, self).__init__()
        self.roberta = base_model.cuda(gpu)
        self.bert = bert_model.cuda(gpu)
        self.emb = self.roberta.embeddings
        self.blocks = self.roberta.encoder.layer

    def forward(self, layer_num, inp, attn_masks):

        if layer_num == 0:
            input_shape = inp.size()
            inp = inp.view(-1, input_shape[-1])
            return self.emb(inp)

        else:
            block = self.blocks[layer_num - 1]
            device = inp.device
            input_shape = inp.size()
            ext_attn_mask: torch.Tensor = self.bert.get_extended_attention_mask(attn_masks, input_shape, device)
            return block(inp, ext_attn_mask)[0]

        # hidden_states = self.ln_f(hidden_states)
        # output_shape = input_shape + (hidden_states.size(-1),)
        # hidden_states = hidden_states.view(*output_shape)

class LayerXlnet(nn.Module):

    def __init__(self, base_model, gpu):
        super(LayerXlnet, self).__init__()
        self.xln = xln = base_model.cuda(gpu)
        self.emb = xln.word_embedding
        self.blocks = xln.layer
        self.dtype = xln.dtype

    def forward(self, layer_num, hidden_h, hidden_g, attention_mask):
        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # so we move here the first dimension (batch) to the end
        hidden_h = hidden_h.transpose(0, 1).contiguous()
        if hidden_g:
          hidden_g = hidden_g.transpose(0, 1).contiguous()
        qlen, bsz = hidden_h.shape[0], hidden_h.shape[1]

        if layer_num == 0:
            inp = hidden_h
            emb = self.emb(inp)
            emb = emb.permute(1, 0, 2).contiguous()
            return emb, None

        else:
            attention_mask = attention_mask.transpose(0, 1).contiguous() if attention_mask is not None else None

            klen = qlen

            dtype_float = self.dtype

            # data mask: input mask & perm mask
            input_mask = 1.0 - attention_mask
            data_mask = input_mask[None]

            attn_mask = data_mask[:, :, :, None]
            attn_mask = (attn_mask > 0).to(dtype_float)
            non_tgt_mask = -torch.eye(qlen).to(attn_mask)
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask)

            # Positional encoding
            pos_emb = self.xln.relative_positional_encoding(qlen, klen, bsz=bsz)
            pos_emb = self.xln.dropout(pos_emb)

            block = self.blocks[layer_num-1]

            outputs = block(
                hidden_h,
                hidden_g,
                attn_mask_h=non_tgt_mask,
                attn_mask_g=attn_mask,
                r=pos_emb,
                seg_mat=None
            )
            output_h, output_g = outputs[:2]

            # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
            output_h = output_h.permute(1, 0, 2).contiguous()
            if output_g:
              output_g = output_g.permute(1, 0, 2).contiguous()

            # output = output_g if output_g is not None else output_h
            return output_h, output_g
