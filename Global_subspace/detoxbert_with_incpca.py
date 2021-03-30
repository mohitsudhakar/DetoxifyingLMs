import numpy as np
import nltk

from Local_debias.utils.data_utils import DataUtils
from model_utils import initBert

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import string
puncs = string.punctuation.replace('*', '').replace('#', '')
table = str.maketrans('', '', puncs)
# !pip install nvsmi
# import nvsmi
import model_utils
import torch
import torch.nn as nn
import torch.nn.functional as F

# !pip install transformers
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
# !pip install livelossplot

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
num_components = batch_size
count = num_sents*num_words
# model = bert_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from sklearn.decomposition import IncrementalPCA
incPca = IncrementalPCA(n_components=num_components)

def run_bert_algorithm_with_inc_pca(t_inputs, nt_inputs, model):
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
  # getMemUtil('run')
  print(torch.Tensor(np.array(incPca.components_)).shape)
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
  # print(t_sents[0])
  # print(nt_sents[0])
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
  # getMemUtil('out1')
  torch.cuda.empty_cache()
  run_bert_algorithm_with_inc_pca(inputs, nt_inputs, model)
  batch_no += 1
  # getMemUtil('out2')
  torch.cuda.empty_cache()

pcs = np.array(incPca.components_)  # pcs.shape = torch.Size([50, 768])
ev_ratio = np.array(incPca.explained_variance_ratio_)
ev = np.array(incPca.explained_variance_)

with open('princComp_top100words_50pc.txt', 'w') as f:
  np.savetxt(f, pcs)

print(batch_no)

import matplotlib.pyplot as plt

def plotVariance(y, title=""):
  x = range(len(y))
  plt.plot(x, y)
  plt.title(title)
  plt.show()
  plt.savefig(title)

plotVariance(ev_ratio, title='EV Ratio')
plotVariance(ev, title='EV Values')

pc_tensor = torch.FloatTensor(pcs)
pc_tensor.shape

"""### Test with one sentence

1. Put sentence through tokenizer and model
2. Call projection() on model output (a) and pc_tensor (b)
3. Output is debiased rep, can be passed downstream

"""

def func(inputs, model):
  model = model.to(device)
  inputs = inputs.to(device)
  outputs = model(**inputs)
  return outputs

sentence = ["What is wrong with you? Is that even possible you fool!"]
inputs = tokenizer(
              sentence, add_special_tokens=True, truncation=True,
              max_length=128, padding='max_length',
              return_attention_mask=True,
              return_tensors='pt')
model_out = func(inputs, model)
print(model_out.last_hidden_state.shape, model_out.pooler_output.shape)
# model_out = model_out.last_hidden_state
model_out = model_out.pooler_output
pc_tensor_ = pc_tensor.to(device)
debiased = model_utils.projection(model_out, pc_tensor_)

print('Model output', model_out.shape)
print('Debiased output', debiased.shape)

# torch.cuda.empty_cache()
# print(list(nvsmi.get_gpus()))

"""**BERT Toxicity Classifier**"""

tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')
bert1 = BertModel.from_pretrained('bert-base-uncased')

class BertClassifierBatch(nn.Module):
  def __init__(self):
    super(BertClassifierBatch, self).__init__()
    self.bert = bert1.to(device)
    self.fc = nn.Linear(768, 2)

  def forward(self, inputs):
    inputs = inputs.to(device)
    out = self.bert(**inputs)
    out = out.pooler_output
    print(out.shape)
    out = self.fc(out)
    return F.softmax(out, dim=1)

cls1 = BertClassifierBatch()
cls1 = cls1.cuda()
# cls1

tokenizer2 = BertTokenizer.from_pretrained('bert-base-uncased')
bert2 = BertModel.from_pretrained('bert-base-uncased')
pc_tensor_ = pc_tensor.to(device)

class DebiasedBertClassifierBatch(nn.Module):
  def __init__(self):
    super(DebiasedBertClassifierBatch, self).__init__()
    self.bert = bert2.to(device)
    self.fc = nn.Linear(768, 2)

  def forward(self, inputs):
    inputs = inputs.to(device)
    out = self.bert(**inputs)
    out = out.pooler_output
    print(out.shape)
    out = model_utils.projection(out, pc_tensor_)
    print(out.shape)
    out = self.fc(out)
    return F.softmax(out, dim=1)

cls2 = DebiasedBertClassifierBatch()
cls2 = cls2.cuda()
# cls2

sentence = ["Hi my name is mark"]
inputs = tokenizer(
              sentence, add_special_tokens=True, truncation=True,
              max_length=128, padding='max_length',
              return_attention_mask=True,
              return_tensors='pt')
out1 = cls1(inputs)
out2 = cls2(inputs)

print('Biased cls output', out1.shape, out1)
print('Debiased cls output', out2.shape, out2)

