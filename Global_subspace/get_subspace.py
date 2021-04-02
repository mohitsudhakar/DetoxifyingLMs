import sys
import numpy as np
import nltk

from Global_classifier.debert_global import BertGlobalClassifier, DeBertGlobalClassifier
from Local_debias.utils.data_utils import DataUtils
from model_utils import initBert

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import string
puncs = string.punctuation.replace('*', '').replace('#', '')
table = str.maketrans('', '', puncs)
# import nvsmi
import model_utils
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# !wget http://cs.virginia.edu/~ms5sw/detox/toxic_sents.txt
# !wget http://cs.virginia.edu/~ms5sw/detox/non_toxic_sents.txt

data = 'data/'
with open(data+'toxic_sents.txt', 'r') as f:
  toxic_sents = f.readlines()
with open(data+'non_toxic_sents.txt', 'r') as f:
  non_toxic_sents = f.readlines()

# todo: Compute PMI

"""**Debiasing BERT**

**TRYING PCA TOGETHER WITH ALL SENTENCES**
"""

seq_len = 128
emb_dim = 768
num_layers = 12
batch_size = 50
num_words = 100
num_sents = 300
num_components = sys.argv[1]
count = num_sents*num_words
# model = bert_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


tokenizer, model = initBert()
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

print('Saving PC matrix to file')

filename = 'princComp_top' + str(num_words) + '_comp' + str(num_components)
with open(filename +  '.txt', 'w') as f:
  np.savetxt(f, pcs)

print('Finished working on', batch_no, 'batches')

import matplotlib.pyplot as plt

def plotVariance(y, title=""):
  x = range(len(y))
  plt.plot(x, y)
  plt.title(title)
  plt.show()
  plt.savefig(title)

plotVariance(ev_ratio, title='EV Ratio_'+filename)
plotVariance(ev, title='EV Values_'+filename)

pc_tensor = torch.FloatTensor(pcs)
print(pc_tensor.shape)
pc_tensor_ = pc_tensor.to(device)


