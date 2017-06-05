import json

import joblib
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, PReLU, BatchNormalization, Dropout
from keras.models import Sequential, load_model
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from models.utils import common_model_parser, calculate_statistics


def add_feature_importance(bst, log_file: str):
    log_data = json.load(open(log_file))
    log_data['feature_importance'] = dict(zip(bst.feature_name(), bst.feature_importance().tolist()))
    json.dump(log_data, open(log_file, 'w'), sort_keys=True, indent=4)


def add_train_score(y_train, p_train, log_file: str, weight):
    log_data = json.load(open(log_file))
    log_data['results']['train_log_loss'] = log_loss(y_train, p_train, sample_weight=weight)
    print(log_data)


# neural net
def nn_model(X_train):
    model = Sequential()

    model.add(Dense(400, input_dim=X_train.shape[1], kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.45))

    model.add(Dense(200, kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.22))

    model.add(Dense(50, kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adadelta')
    return model

def calibration(values):
    epsilon = 1e-7
    return np.minimum(1 - epsilon, np.maximum(epsilon, values))


def main():
    options = common_model_parser().parse_args()
    config = json.load(open(options.config_file))
    data = pd.read_csv(options.data_file)
    feature_columns = data.columns[data.columns != 'y']
    class_weight = {0: 1.309028344, 1: 0.472001959}

    if options.train:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=114514)
        negative_weight = (data.y.sum() / config['model']["target_positive_ratio"] - data.y.sum()) / (data.y == 0).sum()

        scalers = []
        stats = {"results": [], "config": config}
        data['prediction'] = [0] * data.shape[0]
        for i, (train, valid) in enumerate(skf.split(data[feature_columns], data['y'])):
            train_data = data.ix[train]
            valid_data = data.ix[valid]
            X_train, y_train = train_data[feature_columns], train_data['y']
            X_valid, y_valid = valid_data[feature_columns], valid_data['y']

            w_train = np.ones(X_train.shape[0])
            w_train *= 0.472001959
            w_train[y_train == 0] = 1.309028344
            w_valid = np.ones(X_valid.shape[0])
            w_valid *= 0.472001959
            w_valid[y_valid == 0] = 1.309028344

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train.values)
            X_valid = scaler.transform(X_valid.values)
            model = nn_model(X_train)
            bst_model_path = options.model_file + '.h5'
            hist = model.fit(X_train, y_train, validation_data=(X_valid, y_valid, w_valid),
                             epochs=100, batch_size=2048, shuffle=True,
                             class_weight=class_weight, callbacks=[EarlyStopping(monitor='val_loss', patience=3),
                                                                   ModelCheckpoint(bst_model_path, save_best_only=True,
                                                                                   save_weights_only=True)])

            p_valid = calibration(model.predict(X_valid, batch_size=8192, verbose=1).ravel())
            p_train = calibration(model.predict(X_train, batch_size=8192, verbose=1).ravel())
            print(p_valid, p_valid.max())
            joblib.dump(p_valid, 'tmp.p_valid.pkl')
            model.save(options.model_file + '.{}.h5'.format(i))
            scalers.append(scaler)
            data.ix[valid, 'prediction'] = p_valid
            stat = calculate_statistics(pred=p_valid, true=y_valid, weight=w_valid)
            stat['results']['train_log_loss'] = log_loss(y_train, p_train, sample_weight=w_train)
            print(stat)
            stats["results"].append(stat["results"])
        joblib.dump(scalers, options.model_file + '.scaler')
        data[['prediction']].to_csv(options.model_file + '.train.pred', index=False)
        print(stats)
        json.dump(stats, open(options.log_file, 'w'), sort_keys=True, indent=4)
    else:
        scalers = joblib.load(options.model_file + '.scaler')
        data['is_duplicate'] = np.zeros(data.shape[0])
        preds = np.zeros((data.shape[0], len(scalers)))
        for i, scaler in enumerate(scalers):
            model = load_model(options.model_file + '.{}.h5'.format(i))
            X = scaler.transform(data[feature_columns])
            preds[:, i] = calibration(model.predict(X, batch_size=8192, verbose=1).ravel())
        data['is_duplicate'] = preds.mean(axis=1)
        data[['is_duplicate']].to_csv(options.submission_file, index_label='test_id')


if __name__ == "__main__":
    main()
