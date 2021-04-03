import argparse

from Local_debias.utils.data_utils import DataUtils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--num_words", help="Number of words to be considered")
    parser.add_argument("-p", "--data_path", help="Data path, data/")
    args = parser.parse_args()

    num_words = args.num_words
    data_path = args.data_path
    t_path = data_path + 'toxic_sents_'+str(num_words)+'.txt'
    nt_path = data_path + 'non_toxic_sents_'+str(num_words)+'.txt'

    print('Data Preprocessing (4 steps)')

    dataClass = DataUtils()
    print('1. Read file, get df')
    df, toxic_df, nontox_df = dataClass.readToxFile(path=data_path)
    print('2. Get word to sentence dict')
    wsentAll, wsentTox, wsentNT = dataClass.readWordToSentFiles(path=data_path)
    print('3. Get word scores dict')
    sAll, sTox, sNT = dataClass.readWordScores(path=data_path)
    print('4. Process to get final dataframe')
    ht = dataClass.process(sAll, sTox, sNT)

    print('Data preproc done!')

    toxic_words = ht['Word'][:num_words]
    sents = []
    words = []
    for word in toxic_words:
      l = len(wsentTox[word])
      sents.extend(wsentTox[word])
      words.extend([word]*l)

    with open(t_path, 'w') as f:
      for sent in sents:
        f.write(sent + '\n')

    nt_sents = [sents[i].replace(words[i], '[PAD]') for i in range(len(sents))]
    with open(nt_path, 'w') as f:
      for sent in nt_sents:
        f.write(sent + '\n')

    print('Wrote toxic_sents and non_toxic_sents to files.')