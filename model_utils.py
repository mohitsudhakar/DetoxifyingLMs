import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# import nvsmi
from Global_classifier.debert_global import BertGlobalClassifier, DeBertGlobalClassifier
from Global_classifier.degpt_global import GPT2GlobalClassifier, GPT2GlobalClassifier2, DeGPT2GlobalClassifier
from Local_classifier.models.bert import BertClassifier, DeBertClassifier


def plotGraph(y, title=""):
  x = range(len(y))
  plt.plot(x, y)
  plt.title(title)
  plt.show()
  plt.savefig(title)


def plotPC(ev, title, num_pcs=3):
  for pc in range(num_pcs):
    pcs = [f[pc] for f in ev]
    x = range(len(pcs))
    plt.plot(x, pcs)
  plt.legend(['PC'+str(i) for i in range(1, num_pcs+1)])
  plt.savefig(title + '.png')
  plt.close()


# Computes PCs of difference vector
def getPrincipalComponents(D, num_comp=None):
  # print(D.shape)
  pca = PCA(n_components=num_comp, svd_solver="auto")
  X = D.cpu().detach().numpy()
  pca.fit(X)
  ev_percent = pca.explained_variance_ratio_
  ev = pca.explained_variance_
  return torch.Tensor(np.array(pca.components_)), np.array(ev_percent), np.array(ev)


# def getMemUtil(msg=''):
#   print(msg, str(list(nvsmi.get_gpus())[0]).split(' | ')[3])

def projection(a, b):
  inner = torch.mm(a, b.T)
  res = a - torch.mm(inner, b)
  return res


def initBert():
  from transformers import BertTokenizer, BertModel
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased')
  return tokenizer, model


def initGpt2():
  from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
  gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_prefix_space=True)
  gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  gpt2_config = GPT2Config(vocab_size=gpt2_tokenizer.vocab_size)
  gpt2_model = GPT2Model.from_pretrained('gpt2', config=gpt2_config)
  gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
  return gpt2_tokenizer, gpt2_model


def initRoberta():
  from transformers import RobertaTokenizer, RobertaModel, BertModel
  roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
  roberta_model = RobertaModel.from_pretrained('roberta-base')
  bert_model = BertModel.from_pretrained('bert-base-uncased')
  return roberta_tokenizer, roberta_model, bert_model


def initXlnet():
  from transformers import XLNetTokenizer, XLNetModel
  xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
  xlnet_model = XLNetModel.from_pretrained('xlnet-base-cased')
  return xlnet_tokenizer, xlnet_model


def initGlobalBert():
  model = BertGlobalClassifier()
  return model

def initGlobalGpt2():
  model = GPT2GlobalClassifier()
  return model

def initGlobalGpt2_2():
  model = GPT2GlobalClassifier2()
  return model


def initGlobalRoberta():
  return


def initGlobalXlnet():
  return


def initGlobalBertDebiased(subspace):
  model = DeBertGlobalClassifier(subspace)
  return model

def initGlobalGpt2Debiased(subspace):
  model = DeGPT2GlobalClassifier(subspace)
  return model


def initGlobalRobertaDebiased():
  return


def initGlobalXlnetDebiased():
  return


def getPretrained(model):
  switcher = {
    'bert': initBert,
    'gpt2': initGpt2,
    'gpt2_2': initGpt2,
    'roberta': initRoberta,
    'xlnet': initXlnet
  }
  return switcher.get(model, (None, None))()

def getGlobalModel(model_name):
  switcher = {
    'bert': initGlobalBert,
    'gpt2': initGlobalGpt2,
    'gpt2_2': initGlobalGpt2_2,
    'roberta': initGlobalRoberta,
    'xlnet': initGlobalXlnet
  }
  return switcher.get(model_name, (None, None))()



def getGlobalModelDebiased(model_name, subspace):
  switcher = {
    'bert': initGlobalBertDebiased,
    'gpt2': initGlobalGpt2Debiased,
    'roberta': initGlobalRobertaDebiased,
    'xlnet': initGlobalXlnetDebiased
  }
  return switcher.get(model_name, (None, None))(subspace)


def getSwitcher(withDebias = True):
  if withDebias:
    switcher = {
      'bert': DeBertClassifier,
      # 'gpt2': clsGpt2,
      # 'roberta': clsRoberta,
      # 'xlnet': clsXlnet
    }
  else:
    switcher = {
      'bert': BertClassifier,
      # 'gpt2': Gpt2Classifier(),
      # 'roberta': clsRoberta,
      # 'xlnet': clsXlnet
    }
  return switcher


def getClassifier(modelname, withDebias=True):
  return getSwitcher(withDebias).get(modelname, None)()


def getAttentionMask(attention_mask, batch_size, dtype):
  attention_mask = attention_mask.view(batch_size, -1)
  attention_mask = attention_mask[:, None, None, :]
  attention_mask = attention_mask.to(dtype=dtype)  # fp16 compatibility
  attention_mask = (1.0 - attention_mask) * -10000.0
  return attention_mask