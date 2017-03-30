import ast
import os
import sys

from features.transform import nltk_pos_tag
from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix


def noun_jaccard(row):
    q1set = set([s for (s, tag) in ast.literal_eval(row['question1']) if tag.startswith('N')])
    q2set = set([s for (s, tag) in ast.literal_eval(row['question2']) if tag.startswith('N')])
    num = len(q1set.intersection(q2set))
    den = len(q1set.union(q2set))
    return round(float(num) / den if den > 0 else 1, 5)


def create_feature(data_file):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return
    print(sys.argv[0], file=sys.stderr)
    df = nltk_pos_tag(data_file)
    print(df.head(), file=sys.stderr)
    column_name = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])
    df[column_name] = df.apply(noun_jaccard, axis=1, raw=True).values
    df[[column_name]].to_csv(feature_output_file(data_file), index=False, float_format='%.5f')


def main():
    options = common_feature_parser().parse_args()
    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_feature(data_file=file_name)


if __name__ == "__main__":
    main()
