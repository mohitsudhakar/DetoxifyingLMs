import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from Global_classifier.dealbert_global import DeAlbertGlobalClassifier, AlbertGlobalClassifier
from Global_classifier.debert_global import BertGlobalClassifier, DeBertGlobalClassifier
from Global_classifier.degpt_global import GPT2GlobalClassifier, DeGPT2GlobalClassifier
from Global_classifier.deroberta_global import RobertaGlobalClassifier, DeRobertaGlobalClassifier
from Global_classifier.distilbert_global import DistilBertGlobalClassifier, DeDistilBertGlobalClassifier
from Local_classifier.models.bert import BertClassifier, DeBertClassifier


def getPooledOutput(out, model_name, input_ids, tokenizer, batch_size):
  pooled_out = None

  if model_name in ['gpt2']:
    out = out.last_hidden_state
    sequence_lengths = torch.ne(input_ids, tokenizer.pad_token_id).sum(-1) - 1
    pooled_out = out[range(batch_size), sequence_lengths]

  elif model_name in ['bert', 'roberta', 'albert']:
    pooled_out = out.pooler_output

  elif model_name in ['distilbert']:
    hidden_state = out.last_hidden_state  # (bs, seq_len, dim)
    pooled_out = hidden_state[:, 0]  # (bs, dim)
    # pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
    # pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
    # pooled_output = self.dropout(pooled_output)  # (bs, dim)
    # logits = self.fc(pooled_output)  # (bs, num_labels)

  return pooled_out


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


def removeComponent(a, b):
  inner = torch.mm(a, b.T)
  res = a - torch.mm(inner, b)
  return res


def initBert(freeze_weights=False):
  from transformers import BertTokenizer, BertModel
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased')

  if freeze_weights:
    for param in model.parameters():
      param.requires_grad = False

  return tokenizer, model


def initGpt2(freeze_weights=False):
  from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
  gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_prefix_space=True)
  gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  gpt2_config = GPT2Config(vocab_size=gpt2_tokenizer.vocab_size)
  gpt2_model = GPT2Model.from_pretrained('gpt2', config=gpt2_config)
  gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))

  if freeze_weights:
    for param in gpt2_model.parameters():
      param.requires_grad = False

  return gpt2_tokenizer, gpt2_model



def initRoberta(freeze_weights=False):
  from transformers import RobertaTokenizer, RobertaModel, BertModel
  roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
  roberta_model = RobertaModel.from_pretrained('roberta-base')
  bert_model = BertModel.from_pretrained('bert-base-uncased')
  if freeze_weights:
    for param in roberta_model.parameters():
      param.requires_grad = False

  return roberta_tokenizer, roberta_model, bert_model

def initAlbert(freeze_weights=False):
  from transformers import AlbertTokenizer, AlbertModel
  tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
  model = AlbertModel.from_pretrained('albert-base-v2')

  if freeze_weights:
    for param in model.parameters():
      param.requires_grad = False

  return tokenizer, model

def initDistilBert(freeze_weights=False):
  from transformers import DistilBertTokenizer, DistilBertModel
  tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
  model = DistilBertModel.from_pretrained('distilbert-base-uncased')

  if freeze_weights:
    for param in model.parameters():
      param.requires_grad = False

  return tokenizer, model

def initXlnet():
  from transformers import XLNetTokenizer, XLNetModel
  xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
  xlnet_model = XLNetModel.from_pretrained('xlnet-base-cased')
  return xlnet_tokenizer, xlnet_model


def initGlobalBert(freeze_weights=False):
  model = BertGlobalClassifier(freeze_weights)
  return model

def initGlobalGpt2(freeze_weights=False):
  model = GPT2GlobalClassifier(freeze_weights)
  return model


def initGlobalRoberta(freeze_weights=False):
  return RobertaGlobalClassifier(freeze_weights)


def initGlobalAlbert(freeze_weights=False):
  model = AlbertGlobalClassifier(freeze_weights)
  return model

def initGlobalDistilBert(freeze_weights=False):
  model = DistilBertGlobalClassifier(freeze_weights)
  return model

def initGlobalXlnet():
  return


def initGlobalBertDebiased(subspace, freeze_weights=False):
  model = DeBertGlobalClassifier(subspace, freeze_weights)
  return model

def initGlobalGpt2Debiased(subspace, freeze_weights=False):
  model = DeGPT2GlobalClassifier(subspace, freeze_weights)
  return model


def initGlobalRobertaDebiased(subspace, freeze_weights=False):
  return DeRobertaGlobalClassifier(subspace, freeze_weights)


def initGlobalAlbertDebiased(subspace, freeze_weights=False):
  model = DeAlbertGlobalClassifier(subspace, freeze_weights)
  return model

def initGlobalDistilBertDebiased(freeze_weights=False):
  model = DeDistilBertGlobalClassifier(freeze_weights)
  return model


def initGlobalXlnetDebiased():
  return


def getPretrained(model):
  switcher = {
    'bert': initBert,
    'gpt2': initGpt2,
    'gpt2_2': initGpt2,
    'roberta': initRoberta,
    'albert': initAlbert,
    'xlnet': initXlnet
  }
  return switcher.get(model, (None, None))()

def getGlobalModel(model_name, freeze_weights=False):
  switcher = {
    'bert': initGlobalBert,
    'gpt2': initGlobalGpt2,
    'roberta': initGlobalRoberta,
    'albert': initGlobalAlbert,
    'xlnet': initGlobalXlnet
  }
  return switcher.get(model_name, (None, None))(freeze_weights)



def getGlobalModelDebiased(model_name, subspace, freeze_weights=False):
  switcher = {
    'bert': initGlobalBertDebiased,
    'gpt2': initGlobalGpt2Debiased,
    'roberta': initGlobalRobertaDebiased,
    'albert': initGlobalAlbertDebiased,
    'xlnet': initGlobalXlnetDebiased
  }
  return switcher.get(model_name, (None, None))(subspace, freeze_weights)


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