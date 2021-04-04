from torch.utils.data.dataset import Dataset


"""**Build Toxic Dataset**"""
class ToxicityDataset(Dataset):
    def __init__(self, toxic_df, nontox_df, batch_size):

        self.batch_size = batch_size
        self.texts = []
        self.labels = []
        for i, row in toxic_df.iterrows():
            text = row['Text']
            if not text.strip():
                continue
            self.texts.append(text)
            self.labels.append(1)
        for i, row in nontox_df.iterrows():
            text = row['Text']
            if not text.strip():
                continue
            self.texts.append(text)
            self.labels.append(0)

        print('num ones', len(list(filter(lambda l: l == 1, self.labels))))
        print('num zero', len(list(filter(lambda l: l == 0, self.labels))))
        print('total', len(self.labels), len(self.texts))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]
