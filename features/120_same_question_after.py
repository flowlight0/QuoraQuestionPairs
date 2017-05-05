import os

import pandas as pd
from tqdm import tqdm

from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix


def try_apply_dict(x, dict_to_apply):
    try:
        return dict_to_apply[x]
    except KeyError:
        return 0


def create_feature(data_file, df):
    if os.path.exists(feature_output_file(data_file)) and False:
        print('File exists {}.'.format(feature_output_file(data_file)))
        return
    column_name_prefix = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])
    q1_see_later = column_name_prefix + '.q1'
    q2_see_later = column_name_prefix + '.q2'
    column_names = [q1_see_later, q2_see_later]
    out_df = pd.DataFrame()
    out_df[q1_see_later] = df['see_later1'].tolist()
    out_df[q2_see_later] = df['see_later2'].tolist()
    out_df[column_names].to_csv(feature_output_file(data_file), index=False)


def main():
    options = common_feature_parser().parse_args()
    train_file = dict(generate_filename_from_prefix(options.data_prefix))['train']
    test_file = dict(generate_filename_from_prefix(options.data_prefix))['test']

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    test_df['is_duplicate'] = -1
    test_df.rename(columns={'test_id': 'id'}, inplace=True)
    df = train_df.append(test_df)
    df.reset_index(inplace=True)
    from collections import Counter
    see_later1 = []
    see_later2 = []
    sentence_counter1 = Counter()
    sentence_counter2 = Counter()

    for i in tqdm(range(df.shape[0])):
        row = df.iloc[-i - 1]
        q1 = str(row['question1'])
        q2 = str(row['question2'])
        see_later1.append(sentence_counter1[q1])
        see_later2.append(sentence_counter2[q2])
        sentence_counter1[q1] += 1
        sentence_counter2[q2] += 1
    df['see_later1'] = list(reversed(see_later1))
    df['see_later2'] = list(reversed(see_later2))
    create_feature(data_file=train_file, df=df[df.is_duplicate >= 0])
    create_feature(data_file=test_file, df=df[df.is_duplicate < 0])


if __name__ == "__main__":
    main()
