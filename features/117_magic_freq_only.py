import os

import pandas as pd

from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix


def try_apply_dict(x, dict_to_apply):
    try:
        return dict_to_apply[x]
    except KeyError:
        return 0


def create_feature(data_file, questions_dict, q1_vc, q2_vc):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return

    df = pd.read_csv(data_file)
    column_name_prefix = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])
    q1_hash = column_name_prefix + '.q1_hash.cat'
    q2_hash = column_name_prefix + '.q2_hash.cat'
    q1_freq = column_name_prefix + '.q1_freq'
    q2_freq = column_name_prefix + '.q2_freq'

    column_names = [q1_freq, q2_freq]
    df[q1_hash] = df['question1'].map(questions_dict)
    df[q2_hash] = df['question2'].map(questions_dict)
    df[q1_freq] = df[q1_hash].map(lambda x: try_apply_dict(x, q1_vc) + try_apply_dict(x, q2_vc))
    df[q2_freq] = df[q2_hash].map(lambda x: try_apply_dict(x, q1_vc) + try_apply_dict(x, q2_vc))
    df[column_names].to_csv(feature_output_file(data_file), index=False)


def main():
    options = common_feature_parser().parse_args()
    # from https://www.kaggle.com/jturkewitz/magic-features-0-03-gain/notebook
    train_orig = pd.read_csv(dict(generate_filename_from_prefix(options.data_prefix))['train'])
    test_orig = pd.read_csv(dict(generate_filename_from_prefix(options.data_prefix))['test'])

    df1 = train_orig[['question1']].copy()
    df2 = train_orig[['question2']].copy()
    df1_test = test_orig[['question1']].copy()
    df2_test = test_orig[['question2']].copy()

    df2.rename(columns={'question2': 'question1'}, inplace=True)
    df2_test.rename(columns={'question2': 'question1'}, inplace=True)

    train_questions = df1.append(df2)
    train_questions = train_questions.append(df1_test)
    train_questions = train_questions.append(df2_test)
    train_questions.drop_duplicates(subset=['question1'], inplace=True)

    train_questions.reset_index(inplace=True, drop=True)
    questions_dict = pd.Series(train_questions.index.values, index=train_questions.question1.values).to_dict()
    train_cp = train_orig.copy()
    test_cp = test_orig.copy()
    train_cp.drop(['qid1', 'qid2'], axis=1, inplace=True)

    test_cp['is_duplicate'] = -1
    test_cp.rename(columns={'test_id': 'id'}, inplace=True)
    comb = pd.concat([train_cp, test_cp])

    comb['q1_hash'] = comb['question1'].map(questions_dict)
    comb['q2_hash'] = comb['question2'].map(questions_dict)
    q1_vc = comb.q1_hash.value_counts().to_dict()
    q2_vc = comb.q2_hash.value_counts().to_dict()
    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_feature(data_file=file_name, questions_dict=questions_dict, q1_vc=q1_vc, q2_vc=q2_vc)


if __name__ == "__main__":
    main()
