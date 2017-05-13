import os
import sys
import time
from abc import ABCMeta, abstractmethod

import boto3
import joblib
import pandas as pd
from botocore.exceptions import ClientError


class S3LocalCachedCreator(metaclass=ABCMeta):
    def __init__(self, base_dir, feature_name, bucket_name=None, **args):
        self.base_dir = base_dir
        self.feature_name = feature_name
        self.bucket_name = bucket_name
        self.prepare_with_cache_check(**args)

    @abstractmethod
    def create_feature(self, data, local_file=None, s3_file=None):
        pass

    @abstractmethod
    def get_local_cache(self, local_file):
        pass

    @abstractmethod
    def get_s3_cache(self, local_file, s3_file):
        pass

    @abstractmethod
    def read_feature_from_cache(self, local_file):
        pass

    @abstractmethod
    def put_local_cache(self, feature, local_file):
        pass

    @abstractmethod
    def put_s3_cache(self, feature, local_file, s3_file):
        pass

    @abstractmethod
    def calculate_feature(self, data):
        pass

    @abstractmethod
    def prepare_with_cache_check(self, **params):
        pass


class StandardS3LocalCachedCreator(S3LocalCachedCreator):
    def create_feature(self, data, local_file=None, s3_file=None):
        # I assume type(data) is str representing a name of an input data file.
        if local_file is None:
            local_file = self.get_local_file_name(data)
        if s3_file is None:
            s3_file = self.get_s3_file_name(data)

        cached_feature = self.get_local_cache(local_file)
        if cached_feature is not None:
            return cached_feature

        cached_feature = self.get_s3_cache(local_file, s3_file)
        if cached_feature is not None:
            self.put_local_cache(cached_feature, local_file)
            return cached_feature

        print("Started feature calculation ({})".format(self.feature_name))
        start_time = time.time()
        feature = self.calculate_feature(data)
        print("Finished feature calculation ({}): {:.2f}".format(self.feature_name, time.time() - start_time))

        self.put_local_cache(feature, local_file)
        self.put_s3_cache(feature, local_file, s3_file)
        return feature

    def get_local_cache(self, local_file):
        if os.path.exists(local_file):
            print("Local cache exists ({})".format(local_file), file=sys.stderr)
            return self.read_feature_from_cache(local_file)
        else:
            return None

    def get_s3_cache(self, local_file, s3_file):
        try:
            if self.bucket_name is not None:
                s3 = boto3.resource('s3')
                bucket = s3.Bucket(self.bucket_name)
                bucket.download_file(Key=s3_file, Filename=local_file)
                print("S3 cache exists ({})".format(s3_file), file=sys.stderr)
                return self.read_feature_from_cache(local_file)
        except ClientError:
            print("Failed to load s3 cache (bucket={}, key={})".format(self.bucket_name, s3_file), file=sys.stderr)
        return None

    def read_feature_from_cache(self, local_file):
        if local_file.endswith('.csv'):
            return pd.read_csv(local_file)
        else:
            return joblib.load(local_file)

    def put_local_cache(self, feature, local_file):
        if local_file.endswith('.csv'):
            feature.to_csv(local_file, index=False, float_format='%.5f')
        else:
            joblib.dump(feature, local_file)
        print("Saved created feature locally ({})".format(local_file), file=sys.stderr)

    def put_s3_cache(self, feature, local_file, s3_file):
        if self.bucket_name is not None:
            s3 = boto3.resource('s3')
            bucket = s3.Bucket(self.bucket_name)
            bucket.upload_file(Key=s3_file, Filename=local_file)
            print("Saved created feature in s3 ({})".format(s3_file), file=sys.stderr)

    def prepare_with_cache_check(self, train_data, test_data):
        print("Started preparation ({})".format(self.feature_name))
        start_time = time.time()
        need_preparation = False
        for data in [train_data, test_data]:
            local_file = self.get_local_file_name(data)
            s3_file = self.get_s3_file_name(data)
            no_cache = self.get_local_cache(local_file) is None and self.get_s3_cache(local_file, s3_file) is None
            need_preparation = need_preparation or no_cache

        if need_preparation:
            self.prepare_without_cache_check(train_data, test_data)
        print("Finished preparation ({}): {:.2f}".format(self.feature_name, time.time() - start_time))

    @abstractmethod
    def prepare_without_cache_check(self, train_data, test_data):
        pass


    @abstractmethod
    def calculate_feature(self, data):
        pass

    def get_local_file_name(self, data_file):
        return os.path.join(self.base_dir, self.get_s3_file_name(data_file))

    def get_s3_file_name(self, data_file):
        return "data/working/" + os.path.basename(data_file) + "-" + self.feature_name + ".pkl"


