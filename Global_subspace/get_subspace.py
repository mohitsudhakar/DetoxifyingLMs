import argparse
import sys

import numpy as np
import torch

import model_utils
from gpu_utils import getFreeGpu
from model_utils import initBert

# !wget http://cs.virginia.edu/~ms5sw/detox/toxic_sents.txt
# !wget http://cs.virginia.edu/~ms5sw/detox/non_toxic_sents.txt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", help="bert, roberta, gpt2, xlnet")
    parser.add_argument("-n", "--num_components", help="Size of Subspace")
    parser.add_argument("-b", "--batch_size", help="Batch size")
    parser.add_argument("-p", "--data_path", help="Data path, data/")
    parser.add_argument("-w", "--num_words", help="Number of words to be considered")
    args = parser.parse_args()

    seq_len = 128
    emb_dim = 768
    num_layers = 12
    batch_size = int(args.batch_size) if args.batch_size else 50
    num_words = int(args.num_words) if args.num_words else 100
    num_components = int(args.num_components) if args.num_components else 50

    if num_components > batch_size:
        print('Num_components must be lower than batch_size')
        exit(1)

    pc_filename = 'princComp_top_' + str(num_words) + '_comp_' + str(num_components) + '_batch_' + str(batch_size)

    data_path = args.data_path if args.data_path else '../data/'
    with open(data_path+'toxic_sents.txt', 'r') as f:
      toxic_sents = f.readlines()
    with open(data_path+'non_toxic_sents.txt', 'r') as f:
      non_toxic_sents = f.readlines()

    model_name = args.model_name if args.model_name else 'bert'
    tokenizer, model = model_utils.getPretrained(model_name)

    # todo: Compute PMI

    """**Debiasing BERT**
    
    **TRYING PCA TOGETHER WITH ALL SENTENCES**
    """

    # model = bert_model

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(getFreeGpu()))

    from sklearn.decomposition import IncrementalPCA
    incPca = IncrementalPCA(n_components=num_components)

    def run_bert_algorithm_with_inc_pca(t_inputs, nt_inputs, model, device):
      # inputs are encoded sentences
      model = model.to(device)
      t_inputs = t_inputs.to(device)
      nt_inputs = nt_inputs.to(device)

      t_out = model(**t_inputs)
      nt_out = model(**nt_inputs)

      pool_tox = t_out.pooler_output
      pool_ntox = nt_out.pooler_output

      D = pool_tox - pool_ntox
      diff_vector = D.cpu().detach().numpy()
      incPca.partial_fit(diff_vector)
      # print(torch.Tensor(np.array(incPca.components_)).shape)
      torch.cuda.empty_cache()


    l = len(toxic_sents)
    print('Dataset size', l)
    count = 0
    # find toxic words and their sentences
    batch_no = 1
    for i in range(0, l, batch_size):
      print('Batch', batch_no)
      t_sents = toxic_sents[i:i+batch_size]
      nt_sents = non_toxic_sents[i:i+batch_size]
      inputs = tokenizer(
              t_sents, add_special_tokens=True, truncation=True,
              max_length=seq_len, padding='max_length',
              return_attention_mask=True,
              return_tensors='pt')
      nt_inputs = tokenizer(
              nt_sents, add_special_tokens=True, truncation=True,
              max_length=seq_len, padding='max_length',
              return_attention_mask=True,
              return_tensors='pt')
      torch.cuda.empty_cache()
      run_bert_algorithm_with_inc_pca(inputs, nt_inputs, model, device)
      batch_no += 1
      torch.cuda.empty_cache()

    pcs = np.array(incPca.components_)  # pcs.shape = torch.Size([num_components, 768])
    ev_ratio = np.array(incPca.explained_variance_ratio_)
    ev = np.array(incPca.explained_variance_)

    print('Saving PC matrix to file', pc_filename)

    with open(pc_filename +  '.txt', 'w') as f:
      np.savetxt(f, pcs)

    print('Finished working on', batch_no, 'batches')

    import matplotlib.pyplot as plt

    def plotVariance(y, title=""):
      x = range(len(y))
      plt.plot(x, y)
      plt.title(title)
      plt.show()
      plt.savefig(title)

    plotVariance(ev_ratio, title='EV Ratio_'+pc_filename)
    plotVariance(ev, title='EV Values_'+pc_filename)

    # pc_tensor = torch.FloatTensor(pcs)
    # print(pc_tensor.shape)
    # pc_tensor_ = pc_tensor.to(device)


