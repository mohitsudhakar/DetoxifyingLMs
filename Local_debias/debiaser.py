import numpy as np
import torch
from Local_debias.algorithm import sentence_debias, group_debias
import nvsmi

from model_utils import getMemUtil


class Debiaser:

    def __init__(self, df, word_sentence_map, model_name, tokenizer, gpu_list, num_gpus):

        # a = seq_len x 768
        # b = 256 x 768 PC
        # inner = a . bT -> seq x 256
        # a - inner . b -> seq x 768

        self.seq_len = 128
        self.emb_dim = 768
        self.num_layers = 12
        self.batch_size = 1
        self.num_words = 50
        self.num_sents = 100
        self.count = self.num_sents * self.num_words
        self.gpu_list = gpu_list
        self.num_gpus = num_gpus

        self.ht = df
        self.word_sentence_map = word_sentence_map
        self.tokenizer = tokenizer
        self.model_name = model_name

    def processTokensByModel(self, tokens, model_name):
        switcher = {
            'bert': tokens,
            'gpt2': [tok[1:] if tok[0] == 'Ġ' else tok for tok in tokens],
            'xlnet': [tok[1:] if tok[0] == '_' else tok for tok in tokens],
            'roberta': [tok[1:] if tok[0] == 'Ġ' else tok for tok in tokens]
        }
        return switcher.get(model_name, tokens)


    def run(self, model, model_name, debias=True):
        """**Debiasing BERT**"""

        toxic_words = self.ht['Word'][:self.num_words]

        ev_percent = [0]*13
        exp_var = [0]*13
        count = 0
        # find toxic words and their sentences
        for word in toxic_words:
          sents = self.word_sentence_map[word][:self.num_sents]
          print('word', word, 'num_sent', self.num_sents)

          for sent in sents:
            if count % 10 == 0: print(count)
            encoded_text = self.tokenizer.encode_plus(
                sent, add_special_tokens=True, truncation=True,
                max_length=self.seq_len, padding='max_length',
                return_attention_mask=True,
                return_tensors='pt')
            tox, attn_masks = encoded_text['input_ids'], encoded_text['attention_mask']
            ntox = tox.clone()
            tokens = list([self.tokenizer.convert_ids_to_tokens(i) for i in ntox][0])
            tokens = self.processTokensByModel(tokens, model_name)
            try:
              idx = tokens.index(word)
            except:
              print('PROBLEM!')
              print(sent)
              continue

            ntox[0][idx] = 0

            encoded_T = {
                "input_ids": tox,
                "attention_mask": attn_masks
            }
            encoded_NT = {
                "input_ids": ntox,
                "attention_mask": attn_masks
            }
            principal_components, evp, ev = sentence_debias(encoded_T, encoded_NT, model, debias=debias)
            if count % 10 == 0: print('exp_var', np.array(evp).shape)
            for i in range(13):
              ev_percent[i] += evp[i]
              exp_var[i] += ev[i]
            count += 1

        print(count)
        return ev_percent, exp_var


    def run_group(self, model_list, model_name, debias=True):

        toxic_words = self.ht['Word'][:self.num_words]

        count = 0
        enc_tox_list = []
        enc_nontox_list = []
        # find toxic words and their sentences
        for word in toxic_words:

            sents = self.word_sentence_map[word][:self.num_sents]
            print('word', word, 'num_sent', self.num_sents)

            for sent in sents:
                if count % 100 == 0: print(count)
                encoded_text = self.tokenizer.encode_plus(
                    sent, add_special_tokens=True, truncation=True,
                    max_length=self.seq_len, padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt')
                tox, attn_masks = encoded_text['input_ids'], encoded_text['attention_mask']
                ntox = tox.clone()
                tokens = list([self.tokenizer.convert_ids_to_tokens(i) for i in ntox][0])
                tokens = self.processTokensByModel(tokens, model_name)
                try:
                    idx = tokens.index(word)
                except:
                    print('PROBLEM!')
                    print(sent)
                    continue

                ntox[0][idx] = 0

                encoded_T = {
                    "input_ids": tox,
                    "attention_mask": attn_masks
                }
                encoded_NT = {
                    "input_ids": ntox,
                    "attention_mask": attn_masks
                }
                count += 1

                enc_tox_list.append(encoded_T)
                enc_nontox_list.append(encoded_NT)

        print('Cuda :: before run_bert_algo ::', list(nvsmi.get_gpus()))

        per_gpu = count // len(self.gpu_list)
        print('-----------> Per_GPU', per_gpu)
        torch.cuda.empty_cache()
        print('Cuda :: before run_bert_algo ::', list(nvsmi.get_gpus()))

        # gpu = 0
        ev_percent, ev = [0] * self.num_layers, [0] * self.num_layers
        for i in range(0, count, per_gpu):
            # 13 x seq_len
            gpu = i // per_gpu
            print('gpu_it', gpu, 'i', i)
            try:
                evp_, ev_ = group_debias(enc_tox_list[i:i + per_gpu], enc_nontox_list[i:i + per_gpu], per_gpu,
                                        model_list[gpu], debias=debias, gpu=self.gpu_list[gpu], model_name=model_name)
            except:
                print('Problem')
                continue
            evp_ = [e * per_gpu for e in evp_]
            ev_ = [e * per_gpu for e in ev_]
            for l in range(self.num_layers):
                ev_percent[l] += evp_[l]
                ev[l] += ev_[l]

        ev_percent = [e / count for e in ev_percent]
        ev = [e / count for e in ev]
        print(count)

        return ev_percent, ev

    def run_bert_algorithm_with_inc_pca(t_inputs, nt_inputs, model, device, incPca):
        # inputs are encoded sentences
        model = model.to(device)
        t_inputs = t_inputs.to(device)
        nt_inputs = nt_inputs.to(device)
        t_out = model(**t_inputs)
        nt_out = model(**nt_inputs)
        pool_tox = t_out.pooler_output
        pool_ntox = nt_out.pooler_output
        print('Pooled shapes', pool_tox.shape, pool_ntox.shape)

        D = pool_tox - pool_ntox
        print('D shape', D.shape)
        diff_vector = D.cpu().detach().numpy()
        incPca.partial_fit(diff_vector)
        getMemUtil('run')
        print(torch.Tensor(np.array(incPca.components_)).shape)
        torch.cuda.empty_cache()