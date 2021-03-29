import collections
import numpy as np
import pandas as pd
import nltk
# from nltk.stem.porter import PorterStemmer
import string
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch

nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
puncs = string.punctuation.replace('*', '').replace('#', '')
table = str.maketrans('', '', puncs)


def getRTPData(path):
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
    return pd.DataFrame(np.array(rows), columns=columns)


def checks(w):
    if w in stop_words: return False
    if re.search('[a-zA-Z]', w) and '*' in w: return True
    if not w.isalpha(): return False
    if len(w) == 1 and w not in ['a', 'i', 'o', 'u']: return False
    return True


def getWordCounts(texts, scores, word_tokens):
    l = len(texts)
    wordCounts = {}
    for i in range(l):
        if i % 10000 == 0: print(i)
        text, score = texts[i], scores[i]
        wtoks = word_tokens[i]
        for w in wtoks:
            w = w.lower()
            w = w.translate(table)
            if not checks(w): continue
            wordCounts[w] = wordCounts.get(w, 0) + 1
    return wordCounts


def getWordSentences(texts, word_tokens):
    l = len(texts)
    wordSentences = collections.defaultdict(list)
    for i in range(l):
        if i % 10000 == 0: print(i)
        text = texts[i]
        text = text.lower()
        wtoks = word_tokens[i]
        for w in wtoks:
            w = w.lower()
            w = w.translate(table)
            # w = porter.stem(w)
            if not checks(w): continue
            wordSentences[w].append(text)
    return wordSentences


def getSortedWordScores(wordCounts, asc=False):
    rows_list = []
    for i, w in enumerate(wordCounts):
        dic = {'Word': w, 'Count': wordCounts[w]}
        rows_list.append(dic)

    wordScores = pd.DataFrame(rows_list, columns=['Word', 'Count'])

    return wordScores


def preprocess(df, toxic_df, nontox_df, tokenizer):
    texts = df['Text'].to_numpy()
    scores = df['Score'].to_numpy()
    # word_tokens = list(map(word_tokenize, texts))
    encoded_texts = list(map(tokenizer.encode, texts))
    word_tokens = [list([tokenizer.convert_ids_to_tokens(i) for i in encoded_text[1:-1]]) for encoded_text in
                   encoded_texts]
    wcAll = getWordCounts(texts, scores, word_tokens)
    wsentAll = getWordSentences(texts, word_tokens)

    print(encoded_texts[0])
    print(word_tokens[0])

    texts = toxic_df['Text'].to_numpy()
    scores = toxic_df['Score'].to_numpy()
    # word_tokens = list(map(word_tokenize, texts))
    encoded_texts = list(map(tokenizer.encode, texts))
    word_tokens = [list([tokenizer.convert_ids_to_tokens(i) for i in encoded_text[1:-1]]) for encoded_text in
                   encoded_texts]
    wcTox = getWordCounts(texts, scores, word_tokens)
    wsentTox = getWordSentences(texts, word_tokens)

    texts = nontox_df['Text'].to_numpy()
    scores = nontox_df['Score'].to_numpy()
    # word_tokens = list(map(word_tokenize, texts))
    encoded_texts = list(map(tokenizer.encode, texts))
    word_tokens = [list([tokenizer.convert_ids_to_tokens(i) for i in encoded_text[1:-1]]) for encoded_text in
                   encoded_texts]
    wcNT = getWordCounts(texts, scores, word_tokens)
    wsentNT = getWordSentences(texts, word_tokens)

    print(len(wcAll))
    print(len(wcTox))
    print(len(wcNT))

    wordScoresAll = getSortedWordScores(wcAll)
    wordScoresTox = getSortedWordScores(wcTox)
    wordScoresNT = getSortedWordScores(wcNT, asc=True)

    print('All\n', wordScoresAll.head())
    print('Toxic\n', wordScoresTox.head())
    print('NonToxic\n', wordScoresNT.head())

    sAll, sTox, sNT = wordScoresAll, wordScoresTox, wordScoresNT

    from functools import reduce
    dfs = [sAll, sTox, sNT]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on='Word'), dfs)
    df_merged.columns = ['Word', 'Count_All', 'Count_Tox', 'Count_NT']
    df_merged['Tox_percent'] = df_merged.apply(lambda row: row['Count_Tox'] / (row['Count_All']), axis=1)
    df_merged.head()

    MINFREQ = 3
    df_merged_minfreq = df_merged[(df_merged['Count_All'] > MINFREQ)]

    highTox = df_merged_minfreq.sort_values("Tox_percent", ascending=False)
    len(highTox)

    print(highTox[:20])

    ht = highTox[:]

    return ht, wsentTox, wsentNT


def plotVariance(y, title=""):
    x = range(len(y))
    plt.plot(x, y)
    plt.title(title)
    plt.show()
    plt.savefig('plots/' + title)


# Computes PCs of difference vector
def getPrincipalComponents(D, num_comp=None):
  pca = PCA(n_components=num_comp, svd_solver="auto")
  X = D[0].cpu().detach().numpy()
  pca.fit(X)
  exp_var = pca.explained_variance_ratio_
  return torch.Tensor(np.array(pca.components_)), exp_var


def removeProjection(a, b):
  inner = torch.mm(a, b.T)
  res = a - torch.mm(inner, b)
  return res


def plotPCs(ev, title):
    first_pcs = [f[0] for f in ev]
    second_pcs = [f[1] for f in ev]
    third_pcs = [f[2] for f in ev]

    # print('First PCs', first_pcs)
    # plotVariance(first_pcs, title='Debiased BERT - First PC contributions')
    # print('Second PCs', second_pcs)
    # plotVariance(second_pcs, title='Debiased BERT - Second PC contributions')
    # print('Third PCs', third_pcs)
    # plotVariance(third_pcs, title='Debiased BERT - Third PC contributions')

    x = range(len(first_pcs))
    plt.plot(x, first_pcs)
    plt.plot(x, second_pcs)
    plt.plot(x, third_pcs)
    plt.title(title)
    plt.legend(['First PCs', 'Second PCs', 'Third PCs'])
    plt.savefig('plots/'+title+'.png')
    plt.close()

