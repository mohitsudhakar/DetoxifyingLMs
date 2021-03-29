import argparse

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup

from Local_debias.utils.data_utils import DataUtils
from dataset import ToxicityDataset
from model_utils import getPretrained, getClassifier

"""**BERT Toxicity Classifier**"""

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", help="bert, roberta, gpt2, xlnet")
parser.add_argument("-s", "--model_save_name")
parser.add_argument("-d", "--debias", help="Debias, bool")
args = parser.parse_args()

model_name = args.model_name
tokenizer, base_model = getPretrained(model_name)
cls_model = getClassifier(model_name, args.debias)


""" model_save_name = 'model.pt' """
model_save_name = args.model_save_name
path = F"{model_save_name}"
cls_model.load_state_dict(torch.load(path))

cls_model = cls_model.cuda()

batch_size = 32
num_workers = 2
num_epochs = 10

"""**Training Classifier**"""


dataClass = DataUtils(model_name)

df, toxic_df, nontox_df = dataClass.readToxFile()
# wsentAll, wsentTox, wsentNT = dataClass.readWordToSentFiles()
# sAll, sTox, sNT = dataClass.readWordScores()
# ht = dataClass.process(sAll, sTox, sNT)

# Init datasets
dataset = ToxicityDataset(toxic_df=toxic_df, nontox_df=nontox_df, tokenizer=tokenizer, batch_size=batch_size)
dataset_size = len(dataset)
train_size = int(dataset_size*0.8)
val_size = dataset_size - train_size
trainset, valset = random_split(dataset, [train_size, val_size])

def generate_batch(batch):

  texts = [tokenizer.encode_plus(
        entry[0], add_special_tokens=True, truncation=True,
        max_length=128, padding='max_length',
        return_attention_mask=True) for entry in batch] #return_tensors='pt'

  inp_ids = [t['input_ids'] for t in texts]
  inp_ids = torch.LongTensor(inp_ids)
  attn_masks = [t['attention_mask'] for t in texts]
  attn_masks = torch.LongTensor(attn_masks)

  labels = torch.LongTensor([entry[1] for entry in batch])

  return inp_ids, attn_masks, labels

# Init data loaders
train_loader = DataLoader(trainset,
                          batch_size = batch_size,
                          num_workers = num_workers,
                          shuffle = True,
                          collate_fn = generate_batch)
val_loader = DataLoader(valset,
                        batch_size = batch_size,
                        num_workers = num_workers,
                        shuffle = False,
                        collate_fn = generate_batch)

total_steps = len(train_loader) * num_epochs
optimizer = SGD(cls_model.parameters(), lr=0.0001)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
# scheduler = lr_scheduler.CyclicLR(optimizer)
criterion = nn.CrossEntropyLoss()

from livelossplot import PlotLosses

liveloss = PlotLosses()
current_step = 0
best_accuracy = 0
# todo: Add SummaryWriter

(dataset_size, train_size, val_size), len(train_loader), len(val_loader)

for epoch in range(num_epochs):

  print('Epoch', epoch)
  cls_model.train()
  train_loss = 0
  train_acc = 0
  num_samples = 0
  logs = {}

  for batch_id, (inp_ids, attn_masks, labels) in enumerate(train_loader):

    inp_ids = inp_ids.cuda()
    attn_masks = attn_masks.cuda()
    labels = labels.cuda()

    # print('inp_ids', inp_ids.shape, 'attn_masks', attn_masks.shape, 'labels', labels.shape)
    optimizer.zero_grad()

    predicted = cls_model(inp_ids, inp_ids, attn_masks)
    loss = criterion(predicted, labels)
    print('pred', predicted.shape) # batch_size x 2

    train_loss += loss.item()
    loss.backward()
    optimizer.step()
    train_acc += (predicted.argmax(1) == labels).sum().item()
    num_samples += inp_ids.size(0)

    if (1 + batch_id) % 10 == 0:
      print(epoch, batch_id, train_acc / num_samples)
      logs['loss'] = train_loss / num_samples
      logs['accuracy'] = train_acc / num_samples
      liveloss.update(logs)
      liveloss.send()
      current_step += 1

  scheduler.step()

  cls_model.eval()
  val_loss = 0
  val_acc = 0
  num_samples = 0
  for (batch_id, (inp_ids, attn_masks, labels)) in enumerate(val_loader):
    inp_ids = inp_ids.cuda()
    attn_masks = attn_masks.cuda()
    labels = labels.cuda()

    predicted = cls_model(inp_ids, inp_ids, attn_masks)
    loss = criterion(predicted, labels)

    val_loss += loss.item()
    # compute accuracy
    val_acc += (predicted.argmax(1) == labels).sum().item()
    num_samples += inp_ids.size(0)

    if (1 + batch_id) % 10 == 0:
      logs['val_loss'] = val_loss / num_samples
      logs['val_accuracy'] = val_acc / num_samples
      liveloss.update(logs, current_step)
      liveloss.send()


  # Save the parameters for the best accuracy on the validation set so far.
  if logs['val_accuracy'] > best_accuracy:
    best_accuracy = logs['val_accuracy']
    model_save_name = 'toxic_normal_cls.pt'
    path = F"/content/drive/My Drive/Models/{model_save_name}"
    torch.save(cls_model.state_dict(), path)

# !nvidia-smi

# model_save_name = 'toxic_normal_cls.pt'
# path = F"/content/gdrive/My Drive/Models/{model_save_name}"
# model.load_state_dict(torch.load(path))

