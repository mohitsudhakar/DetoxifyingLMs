import os
import json
import collections

def main():
    texts, scores = list(), collections.defaultdict(list)
    metrics = ['Profanity', 'Sexually_Exp', 'Identity_Attack', 'Flirtation', 'Threat', 'Insult', 'Sev_Tox', 'Tox']

    with open('realtoxicityprompts-data/prompts_cleaned.jsonl') as f:
        i=0
        for line in f.readlines():
            i+=1
            if i%1000 == 0:
                print(i)
            data = json.loads(line)
            text = data['prompt']['text'].strip() + ' ' + data['continuation']['text'].strip()
            if i == 1:
                print(data)
            prof = [data['prompt']['profanity'], data['continuation']['profanity']]
            sex = [data['prompt']['sexually_explicit'], data['continuation']['sexually_explicit']]
            identity = [data['prompt']['identity_attack'], data['continuation']['identity_attack']]
            flirt = [data['prompt']['flirtation'], data['continuation']['flirtation']]
            threat = [data['prompt']['threat'], data['continuation']['threat']]
            insult = [data['prompt']['insult'], data['continuation']['insult']]
            severe = [data['prompt']['severe_toxicity'], data['continuation']['severe_toxicity']]
            tox = [data['prompt']['toxicity'], data['continuation']['toxicity']]

            texts.append(text)
            metric_scores = [prof, sex, identity, flirt, threat, insult, severe, tox]
            for i in range(len(metrics)):
                metric = metrics[i]
                score = metric_scores[i]
                if score[0] is None: score[0] = 0
                if score[1] is None: score[1] = 0
                scores[metric].append(float(score[0]) + float(score[1]))

    l = len(texts)
    print(l)
    with open('toxic_sum2.tsv', 'a') as f:
        s = 'Text\tScore'
        # s += '\t'.join(metrics)

        f.write(s + '\n')
        for i in range(l):
            s = texts[i]

            s += '\t' + str(sum([scores[m][i] for m in metrics]))
            # for m in metrics:
            #     s += '\t' + str(scores[m][i])

            f.write(s + '\n')


if __name__ == '__main__':
    main()