# def main():
#     df = pd.read_csv('../data/input/sample_train.csv')
#     nlp = StanfordTokenizer('../corenlp/stanford-corenlp-3.7.0.jar')
#     print(nlp.tokenize(' XXXXXX '.join(df.question1.astype(str).tolist()[:1000])))
from tqdm import tqdm

if __name__ == '__main__':
    # main()
    import spacy
    import pandas as pd
    df = pd.read_csv('../data/input/sample_train.csv')
    # nlp = spacy.load('en_core_web_md')
    nlp = spacy.load('en')

    values = []
    for i, doc in tqdm(enumerate(nlp.pipe(df.question1.tolist(), batch_size=50, n_threads=4))):
        values.append(doc.ents)
    df['question1'] = values

    values = []
    for i, doc in tqdm(enumerate(nlp.pipe(df.question2.tolist(), batch_size=50, n_threads=4))):
        values.append(doc.ents)
    df['question2'] = values
    df.to_csv('tmp.csv', index=False)