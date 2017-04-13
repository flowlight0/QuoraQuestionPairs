import ast
import os
import sys

from features.transform import nltk_pos_tag
from features.utils import feature_output_file, common_feature_parser, generate_filename_from_prefix


def create_feature(data_file):
    if os.path.exists(feature_output_file(data_file)):
        print('File exists {}.'.format(feature_output_file(data_file)))
        return

    df = nltk_pos_tag(data_file)
    print(sys.argv[0], data_file, file=sys.stderr)
    column_name_prefix = 'f{0}'.format(os.path.basename(feature_output_file(data_file)).split('_')[0])

    q1_past = []
    q2_past = []
    for i, row in df.iterrows():
        q1_has = 0
        for w, t in ast.literal_eval(row['question1']):
            if t in ['VBD']:
                q1_has = 1
                break
        q1_past.append(q1_has)

        q2_has = 0
        for w, t in ast.literal_eval(row['question2']):
            if t in ['VBD']:
                q2_has = 1
                break
        q2_past.append(q2_has)
    column_names = [column_name_prefix + '.1', column_name_prefix + '.2']
    df[column_names[0]] = q1_past
    df[column_names[1]] = q2_past
    df[column_names].to_csv(feature_output_file(data_file), index=False, float_format='%.5f')


def main():
    options = common_feature_parser().parse_args()
    for k, file_name in generate_filename_from_prefix(options.data_prefix):
        create_feature(data_file=file_name)


if __name__ == "__main__":
    main()
