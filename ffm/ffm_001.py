from ffmutil import create_ffm_file, train, predict, FFMCV


fields = [
    '1grams_stem_q1',
    '1grams_stem_q2',
    '2grams_stem_q1',
    '2grams_stem_q2',
    '3grams_stem_q1',
    '3grams_stem_q2',
    '4grams_stem_q1',
    '4grams_stem_q2',
    '5grams_stem_q1',
    '5grams_stem_q2',
]


def main():
    ffmcv = FFMCV('ffm_001')
    ffmcv.train(fields)

if __name__ == '__main__':
    main()
