

import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import os
import numpy as np 
from collections import defaultdict
from trainset import Trainset

from Split import train_test_split

class Dataset:
    """ Base class for loading datasets. """

    def __init__(self,reader,rating_scale=(1, 5)):
        self.reader = reader
        self.rating_scale = rating_scale

    @classmethod
    def load_from_file(cls,file_path,reader):
        return DatasetAutoFolds(ratings_file=file_path, reader=reader)
    
    def read_ratings(self, file_name):
        df = self.reader(file_name)
        raw_ratings = df.values
        return raw_ratings
    
    def construct_trainset(self, raw_trainset):

        raw2inner_id_users = {}
        raw2inner_id_items = {}

        current_u_index = 0
        current_i_index = 0

        ur = defaultdict(list)
        ir = defaultdict(list)

        # user raw id, item raw id, translated rating, time stamp
        for urid, irid, r, timestamp in raw_trainset:
            try:
                uid = raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                raw2inner_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                raw2inner_id_items[irid] = current_i_index
                current_i_index += 1

            ur[uid].append((iid, r))
            ir[iid].append((uid, r))

        n_users = len(ur)  # number of users
        n_items = len(ir)  # number of items
        n_ratings = len(raw_trainset)

        trainset = Trainset(
            ur,
            ir,
            n_users,
            n_items,
            n_ratings,
            self.rating_scale,
            raw2inner_id_users,
            raw2inner_id_items,
        )

        return trainset

    def construct_testset(self, raw_testset):

        return [(ruid, riid, r_ui_trans) for (ruid, riid, r_ui_trans, _) in raw_testset]



class DatasetAutoFolds(Dataset):

    def __init__(self, ratings_file=None, reader=None):

        Dataset.__init__(self, reader)
        self.has_been_split = False  # flag indicating if split() was called.

        if ratings_file is not None:
            self.ratings_file = ratings_file
            self.raw_ratings = self.read_ratings(self.ratings_file)
        else:
            raise ValueError("Must specify ratings file or dataframe.")

    def build_full_trainset(self):
        return self.construct_trainset(self.raw_ratings)




if __name__ == '__main__':
    data_storage = 'Datasets'
    file_name = 'MovieLens_100k.txt'

    # get current working directory
    cwd = os.getcwd()
    file_path = os.path.join(cwd,data_storage,file_name)


    def MovieLens_reader(file_path):
        column_names = ['user id','movie id','rating','timestamp']
        df = pd.read_csv(file_path,sep='\t',header=None,names=column_names)
        return df
    reader = MovieLens_reader

    data = Dataset.load_from_file(file_path,reader)


    # split dataset to test and train
    trainset,testset = train_test_split(data,test_size=0.1)
    


