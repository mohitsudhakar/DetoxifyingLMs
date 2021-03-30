import pandas as pd
import numpy as np
import json
from functools import reduce


class DataUtils:

    def __init__(self, model):
        self.path = 'data/'
        self.datapath = self.path + model + '/'

    def readToxFile(self, path=None):
        if not path:
            path = self.path + 'toxic.tsv'
        rows = []
        with open(path, 'r') as f:
            lines = f.readlines()
            columns = lines[0].split('\t')
            num_cols = len(columns)
            lines = lines[1:]
            for line in lines:
                fields = line.split('\t')
                if len(fields) > num_cols:
                    fields = [' '.join(fields[:-num_cols])] + fields[-num_cols:]
                elif len(fields) < num_cols:
                    for i in range(num_cols - len(fields)):
                        fields.append('0')
                rows.append(fields)

        df = pd.DataFrame(np.array(rows), columns=columns)
        # only check toxicity values, todo: check others as well
        # df = df[['Text', 'Tox']]
        df = df[['Text', 'Sev_Tox']]
        df.columns = ['Text', 'Score']
        df.Score = df.Score.astype(float)
        df = df.fillna(0)
        toxic_df = df[df['Score'] >= 0.5]
        nontox_df = df[df['Score'] < 0.5]
        print('done: got toxic data >=0.5')

        return df, toxic_df, nontox_df

    def readWordToSentFiles(self, path=None):
        if not path:
            path = self.datapath
        with open(path+'wsentAll.json') as f1:
            wsentAll = json.load(f1)
        with open(path+'wsentTox.json') as f2:
            wsentTox = json.load(f2)
        with open(path+'wsentNT.json') as f3:
            wsentNT = json.load(f3)

        return wsentAll, wsentTox, wsentNT

    def readWordScores(self, path=None):
        if not path:
            path = self.datapath

        wordScoresAll = pd.read_pickle(path + 'wordScoresAll.pkl')
        wordScoresTox = pd.read_pickle(path + 'wordScoresTox.pkl')
        wordScoresNT = pd.read_pickle(path + 'wordScoresNT.pkl')

        print('All\n', wordScoresAll.head())
        print('Toxic\n', wordScoresTox.head())
        print('NonToxic\n', wordScoresNT.head())

        return wordScoresAll, wordScoresTox, wordScoresNT

    def process(self, sAll, sTox, sNT):

        dfs = [sAll, sTox, sNT]
        df_merged = reduce(lambda left, right: pd.merge(left, right, on='Word'), dfs)

        df_merged.columns = ['Word', 'Count_All', 'Count_Tox', 'Count_NT']
        df_merged.head()

        df_merged['Tox_percent'] = df_merged.apply(lambda row: row['Count_Tox'] / (row['Count_All']), axis=1)

        MINFREQ = 3
        df_merged_minfreq = df_merged[(df_merged['Count_All'] > MINFREQ)]

        # highTox = df_merged_minfreq.sort_values(["Count_All", "Tox_percent"], ascending = [False, False])

        # highTox = df_merged_minfreq.sort_values(["Count_NT", "Count_Tox"], ascending = [True, False])
        highTox = df_merged_minfreq.sort_values("Tox_percent", ascending=False)
        len(highTox)

        highTox[:20]
        return highTox
