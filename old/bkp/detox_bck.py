# Imports

from transformers import BertTokenizer, BertModel

from code.models.layerbert import DeLayerBERT
from code.debias import calcPCAfterDebias

from old.bkp.algoutils_bck import getRTPData, plotPCs, preprocess

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Hyperparams
num_words = 1
num_sents = 10


#!wget http://cs.virginia.edu/~ms5sw/toxic.tsv

path = 'code/data/toxic.tsv'
df = getRTPData(path)
print('done: read file')

# only check toxicity values, todo: check others as well
# df = df[['Text', 'Tox']]
df = df[['Text', 'Sev_Tox']]
df.columns = ['Text', 'Score']
df.Score = df.Score.astype(float)
df = df.fillna(0)
# df = df[:10000]

toxic_df = df[df['Score'] >= 0.5]
nontox_df = df[df['Score'] < 0.5]
print('done: got toxic data >=0.5')

ht, wsentTox, wsentNT = preprocess(df, toxic_df, nontox_df, tokenizer)

# todo: Compute PMI

"""**Debiasing BERT**"""
print("**Debiasing BERT**")

debiasedBert = DeLayerBERT(bert_model, debias=True)
debiasedBert = debiasedBert.cuda()
debiasedBert.eval()


toxic_words = ht['Word'][:num_words]

ev = calcPCAfterDebias(toxic_words=toxic_words,
                              wsentTox=wsentTox,
                              num_sents=num_sents,
                              tokenizer=tokenizer,
                              model=debiasedBert)

plotPCs(ev, "Debiased BERT")

# a = num_sent x 768
# b = 2 x 768 PCs
# inner = a . bT -> num_sent x 2
# inner . b -> num_sent x 768

"""**Normal BERT**"""

biasedBert = DeLayerBERT(bert_model, debias=False)
biasedBert = biasedBert.cuda()
ev = calcPCAfterDebias(toxic_words=toxic_words,
                              wsentTox=wsentTox,
                              num_sents=num_sents,
                              tokenizer=tokenizer,
                              model=biasedBert)

plotPCs(ev, "Normal BERT")

# Toxic
print('top 20 tox%', ht[:20])

# Nontoxic
print('bottom 20 tox%', ht[-20:])
