import os
import sys

import pandas as pd

from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix


def calc_feature(row):
    q1 = str(row['question1']).lower()
    q2 = str(row['question2']).lower()
    c1 = len([word for word in q1.split(' ') if word.isdigit()])
    c2 = len([word for word in q2.split(' ') if word.isdigit()])
    return min(2, abs(c1 - c2))


def create_feature(data_file):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return
    df = pd.read_csv(data_file)
    print(sys.argv[0], data_file, file=sys.stderr)
    column_name = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])
    df[column_name] = df.apply(calc_feature, axis=1, raw=True)
    df[[column_name]].to_csv(feature_output_file(data_file), index=False, float_format='%.5f')


def main():
    options = common_feature_parser().parse_args()
    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_feature(data_file=file_name)


if __name__ == "__main__":
    main()
