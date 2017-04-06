# QuoraQuestionPairs
https://www.kaggle.com/c/quora-question-pairs/team

## Setup

* Download train.csv and test.csv and put them in data/input directory. 
* Setup python3 virtual environment
* Install python packages required in this repository by running `pip install -r required_packages.txt`
* Download some curpus of nltk by running nltk.download() in your python interpreter (I forgot which corpus is required...)

## How to run a model
* Run `python3 run.py --config_file config/(your config file)` (please see existing configuration files)
  * First, it creates all features specified by 'features' in a configuration file and train a model specified by 'model'. It will take about a day to compute all existing features from scratch, but we can skip this step since each feature is cached after its computation. 
  * Then, it trains a model with 80% of training data, evaluate its performance with another 20% of training data, and output statistics and trained model in data/output directory. 
  * Finally, it makes prediction for test data and provides a submission file corresponding to a configuration file. 
