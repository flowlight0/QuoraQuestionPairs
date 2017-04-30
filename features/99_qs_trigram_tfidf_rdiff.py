from features.n_gram_tfidf_relative_difference_base import NGramTfidfRelativeDifference
from features.utils import common_feature_parser


def main():
    parser = common_feature_parser()
    options = parser.parse_args()
    feature_creator = NGramTfidfRelativeDifference(options, ngram_range=(3, 3))
    feature_creator.create()


if __name__ == "__main__":
    main()
