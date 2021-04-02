""" Debiased BERT Toxicity Classifier """

import argparse

import numpy as np
import torch
from livelossplot import PlotLosses
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

import model_utils
from Global_classifier.debert_global import DeBertGlobalClassifier
from Local_debias.utils.data_utils import DataUtils
from dataset import ToxicityDataset
from gpu_utils import getFreeGpu

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", help="bert, roberta, gpt2, xlnet")
    parser.add_argument("-s", "--model_save_name")
    parser.add_argument("-d", "--debias", dest="debias", help="Debias, bool", action="store_true")
    parser.add_argument("-p", "--data_path", help="Data path, data/")
    parser.add_argument("-f", "--subspace_file", help="Subspace matrix filename")
    args = parser.parse_args()
    if not args.subspace_file:
        print("Subspace file required for running Debert!")
        exit(1)

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/debert_global_cls')

    model_name = args.model_name if args.model_name else 'bert'
    tokenizer, base_model = model_utils.getPretrained(model_name)

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    if torch.cuda.is_available():
      device = torch.device('cuda:' + str(getFreeGpu()))

    print('Device', device)

    data_path = args.data_path if args.data_path else '../data/'
    pc_file = args.subspace_file

    loaded_list = np.loadtxt(pc_file, delimiter=" ")
    pcs = torch.FloatTensor(loaded_list)
    pcs = pcs.to(device)
    print('pcs shape', pcs.shape)

    cls_model = DeBertGlobalClassifier(bias_subspace=pcs)
    # cls_model = nn.DataParallel(cls_model)
    cls_model = cls_model.to(device)

    model_save_name = args.model_save_name if args.model_save_name else 'debertG.pt'
    # path = F"{model_save_name}"
    # cls_model.load_state_dict(torch.load(path))

    batch_size = 32
    num_workers = 2
    num_epochs = 10
    step_size = 100

    print('Data Preprocessing (4 steps)')

    dataClass = DataUtils(model_name)
    print('1. Read file, get df')
    df, toxic_df, nontox_df = dataClass.readToxFile(path=data_path)
    print('2. Get word to sentence dict')
    wsentAll, wsentTox, wsentNT = dataClass.readWordToSentFiles(path=data_path)
    print('3. Get word scores dict')
    sAll, sTox, sNT = dataClass.readWordScores(path=data_path)
    # print('4. Process to get final dataframe')
    # ht = dataClass.process(sAll, sTox, sNT)

    print('Creating train and valiation sets')

    validation_split = 0.2
    shuffle = True
    random_seed = 42

    # Init datasets
    dataset = ToxicityDataset(toxic_df=toxic_df, nontox_df=nontox_df, tokenizer=tokenizer, batch_size=batch_size)
    dataset_size = len(dataset)

    #######
    indices = list(range(dataset_size))
    train_indices, test_indices = train_test_split(indices, test_size=validation_split, stratify=dataset.labels)
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)


    #######

    # val_size = dataset_size - train_size
    # print('train size', train_size, 'val size', val_size)
    # trainset, valset = random_split(dataset, [train_size, val_size])

    def generate_batch(batch):

        texts = [tokenizer.encode_plus(
            entry[0], add_special_tokens=True, truncation=True,
            max_length=128, padding='max_length',
            return_attention_mask=True) for entry in batch]  # return_tensors='pt'

        inp_ids = [t['input_ids'] for t in texts]
        inp_ids = torch.LongTensor(inp_ids)
        attn_masks = [t['attention_mask'] for t in texts]
        attn_masks = torch.LongTensor(attn_masks)

        labels = torch.LongTensor([entry[1] for entry in batch])

        return inp_ids, attn_masks, labels


    # Init data loaders
    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              collate_fn=generate_batch,
                              sampler=train_sampler)
    val_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=generate_batch,
                            sampler=test_sampler)

    print('Data Loaders ready')

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
    print(dataset_size, len(train_loader), len(val_loader))

    print('Starting training')
    running_train_loss = 0.0
    running_val_loss = 0.0
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

        # print('inp_ids', inp_ids.shape, 'attn_masks', attn_masks.shape, 'labels', labels.shape)
        optimizer.zero_grad()

        predicted = cls_model(inp_ids, attn_masks)
        loss = criterion(predicted, labels)
        # print('pred', predicted.shape, predicted[0]) # batch_size x 2
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        train_acc += (predicted.argmax(1) == labels).sum().item()
        num_samples += inp_ids.size(0)

        if (1 + batch_id) % step_size == 0:
          print(epoch, batch_id, train_acc / num_samples)
          writer.add_scalar('training loss',
                            train_loss / num_samples,
                            epoch * len(train_loader) + batch_id)
          writer.add_scalar('training accuracy',
                            train_acc / num_samples,
                            epoch * len(train_loader) + batch_id)



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
        # print('val loss', loss)
        val_loss += loss.item()
        # compute accuracy
        val_acc += (predicted.argmax(1) == labels).sum().item()
        num_samples += inp_ids.size(0)

        if (1 + batch_id) % step_size == 0:
            writer.add_scalar('validation loss',
                              val_loss / num_samples,
                              epoch * len(val_loader) + batch_id)
            writer.add_scalar('validation accuracy',
                              val_acc / num_samples,
                              epoch * len(val_loader) + batch_id)

      # Save the parameters for the best accuracy on the validation set so far.
      if val_acc / num_samples > best_accuracy:
        best_accuracy = val_acc / num_samples
        path = F"{model_save_name}"
        torch.save(cls_model.state_dict(), path)

    # !nvidia-smi

    # model_save_name = 'toxic_normal_cls.pt'
    # path = F"/content/gdrive/My Drive/Models/{model_save_name}"
    # model.load_state_dict(torch.load(path))

