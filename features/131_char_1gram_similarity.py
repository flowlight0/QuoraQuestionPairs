from features.char_n_gram_feature_creator import CharNGramSimilarityFeatureCreator
from features.utils import common_feature_parser


def main():
    parser = common_feature_parser()
    options = parser.parse_args()
    feature_creator = CharNGramSimilarityFeatureCreator(options, ngram_range=(1, 1))
    feature_creator.create()


if __name__ == "__main__":
    main()
