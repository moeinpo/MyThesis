

from math import ceil, floor
import numpy as np
from utils import get_rng
from itertools import chain
import numbers


class KFold:
    """ A basic cross-validation iterator.
    Each fold is used once as a testset while the k - 1 remaining folds are
    used for training.
    """

    def __init__(self, n_splits=5, random_state=None, shuffle=True):

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, data):
        """Generator function to iterate over trainsets and testsets.

        Args:
            data: The data containing ratings that will be divided into trainsets and testsets.

        Yields:
            tuple of (trainset, testset)
        """

        if self.n_splits > len(data.raw_ratings) or self.n_splits < 2:
            raise ValueError(
                "Incorrect value for n_splits={}. "
                "Must be >=2 and less than the number "
                "of ratings".format(len(data.raw_ratings))
            )

        # We use indices to avoid shuffling the original data.raw_ratings list.
        indices = np.arange(len(data.raw_ratings))

        if self.shuffle:
            get_rng(self.random_state).shuffle(indices)

        start, stop = 0, 0
        for fold_i in range(self.n_splits):
            start = stop
            stop += len(indices) // self.n_splits
            if fold_i < len(indices) % self.n_splits:
                stop += 1

            raw_trainset = [
                data.raw_ratings[i] for i in chain(indices[:start], indices[stop:])
            ]
            raw_testset = [data.raw_ratings[i] for i in indices[start:stop]]

            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset

    def get_n_folds(self):

        return self.n_splits


def get_cv(cv):
    """Return a 'validated' CV iterator."""

    if cv is None:
        return KFold(n_splits=5)
    if isinstance(cv, numbers.Integral):
        return KFold(n_splits=cv)
    if hasattr(cv, "split") and not isinstance(cv, str):
        return cv  # str have split

    raise ValueError(
        "Wrong CV object. Expecting None, an int or CV iterator, "
        "got a {}".format(type(cv))
    )



class ShuffledSplit:

    """A basic cross-validation iterator with random trainsets and testsets. """

    def __init__(
        self,
        n_splits=5,
        test_size=0.2,
        random_state=None,
        shuffle=True,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle

    def get_valid_test_train_sizes(self,test_size,n_ratings):
        if np.asarray(test_size).dtype.kind == "f":
            test_size = ceil(test_size * n_ratings)        
        train_size = n_ratings - test_size
        return (int(test_size),int(train_size))
    
    def split(self,data):
        n_ratings =len(data.raw_ratings)
        test_size,train_size = self.get_valid_test_train_sizes(self.test_size,n_ratings)

        rng = np.random.RandomState(self.random_state)
        for fold in range(self.n_splits):
            if self.shuffle:
                permutation = rng.permutation(len(data.raw_ratings))
            else:
                permutation = np.arange(len(data.raw_ratings))

            raw_testset = data.raw_ratings[permutation[:test_size]]
            raw_trainset = data.raw_ratings[permutation[test_size : (test_size + train_size)]]

            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset


def train_test_split(data,test_size=0.1,random_state=None, shuffle=True):
    shuffeld_split_instance = ShuffledSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle)
    return next(shuffeld_split_instance.split(data))