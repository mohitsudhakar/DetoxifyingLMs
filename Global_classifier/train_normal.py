""" Normal BERT Toxicity Classifier """

import argparse

import torch
from livelossplot import PlotLosses
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

import model_utils
from Global_classifier.debert_global import BertGlobalClassifier
from Local_debias.utils.data_utils import DataUtils
from dataset import ToxicityDataset
from gpu_utils import getFreeGpu

if __name__ == '__main__':


  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model_name", help="bert, roberta, gpt2, xlnet")
  parser.add_argument("-s", "--model_save_name")
  parser.add_argument("-d", "--debias", dest="debias", help="Debias, bool", action="store_true")
  parser.add_argument("-p", "--data_path", help="Data path, data/")
  args = parser.parse_args()

  writer = SummaryWriter('runs/bert_global_cls')

  model_name = args.model_name if args.model_name else 'bert'
  tokenizer, base_model = model_utils.getPretrained(model_name)

  device = torch.device('cpu')
  if torch.cuda.is_available():
    device = torch.device('cuda' + str(getFreeGpu()))

  print('Device', device)

  data_path = args.data_path if args.data_path else '../data/'

  cls_model = BertGlobalClassifier()

  """ model_save_name = 'model.pt' """
  model_save_name = args.model_save_name if args.model_save_name else 'bertG.pt'
  # path = F"{model_save_name}"
  # cls_model.load_state_dict(torch.load(path))

  cls_model = cls_model.to(device)

  batch_size = 32
  num_workers = 2
  num_epochs = 10

  """**Training Classifier**"""

  print('Data Preprocessing (4 steps)')

  dataClass = DataUtils(model_name)

  df, toxic_df, nontox_df = dataClass.readToxFile(path=data_path)
  print('step1')
  wsentAll, wsentTox, wsentNT = dataClass.readWordToSentFiles(path=data_path)
  print('step2')
  sAll, sTox, sNT = dataClass.readWordScores(path=data_path)
  print('step3')
  ht = dataClass.process(sAll, sTox, sNT)
  print('step4')

  # Init datasets
  dataset = ToxicityDataset(toxic_df=toxic_df, nontox_df=nontox_df, tokenizer=tokenizer, batch_size=batch_size)
  dataset_size = len(dataset)
  train_size = int(dataset_size*0.8)
  val_size = dataset_size - train_size
  print('train size', train_size, 'val size', val_size)
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

  print('data loaders ready')

  total_steps = len(train_loader) * num_epochs
  optimizer = SGD(cls_model.parameters(), lr=0.0001)
  scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps = 0,
                                              num_training_steps = total_steps)
  # scheduler = lr_scheduler.CyclicLR(optimizer)
  criterion = nn.CrossEntropyLoss()

  liveloss = PlotLosses()
  current_step = 0
  best_accuracy = 0
  # todo: Add SummaryWriter

  print('dataset size, train, val')
  print((dataset_size, train_size, val_size), len(train_loader), len(val_loader))

  print('Starting training')

  for epoch in range(num_epochs):

    print('Epoch', epoch)
    cls_model.train()
    train_loss = 0
    train_acc = 0
    num_samples = 0
    logs = {}

    for batch_id, (inp_ids, attn_masks, labels) in enumerate(train_loader):

      inp_ids = inp_ids.to(device)
      attn_masks = attn_masks.to(device)
      labels = labels.to(device)

      print('inp_ids', inp_ids.shape, 'attn_masks', attn_masks.shape, 'labels', labels.shape)
      optimizer.zero_grad()

      predicted = cls_model(inp_ids, attn_masks)
      loss = criterion(predicted, labels)
      # print('pred', predicted.shape, predicted[0]) # batch_size x 2
      print('loss', loss)
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

      inp_ids = inp_ids.to(device)
      attn_masks = attn_masks.to(device)
      labels = labels.to(device)

      predicted = cls_model(inp_ids, attn_masks)
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
      path = F"{model_save_name}"
      torch.save(cls_model.state_dict(), path)

  # !nvidia-smi

  # model_save_name = 'toxic_normal_cls.pt'
  # path = F"/content/gdrive/My Drive/Models/{model_save_name}"
  # model.load_state_dict(torch.load(path))

