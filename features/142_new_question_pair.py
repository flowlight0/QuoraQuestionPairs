import os
from collections import Counter

import pandas as pd
from tqdm import tqdm

from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix


def try_apply_dict(x, dict_to_apply):
    try:
        return dict_to_apply[x]
    except KeyError:
        return 0


def create_feature(data_file, features: pd.DataFrame):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return
    features.to_csv(feature_output_file(data_file), index=False)


def calculate_features(df, counter):
    values = []
    feature_id = 'f' + os.path.basename(str(__file__)).split("_")[0]
    for i, row in tqdm(df.iterrows()):
        values.append([1 if (counter[row['question1']] == 0 and counter[row['question2']] == 0) else 0])
        counter[row['question1']] += 1
        counter[row['question1']] += 1
    return pd.DataFrame(data=values, columns=[feature_id])


def main():
    options = common_feature_parser().parse_args()
    # from https://www.kaggle.com/jturkewitz/magic-features-0-03-gain/notebook
    train_df = pd.read_csv(dict(generate_filename_from_prefix(options.data_prefix))['train'])
    test_df = pd.read_csv(dict(generate_filename_from_prefix(options.data_prefix))['test'])
    counter = Counter()
    features = {'train': calculate_features(train_df, counter), 'test': calculate_features(test_df, counter)}

    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_feature(data_file=file_name, features=features[k])


if __name__ == "__main__":
    main()
