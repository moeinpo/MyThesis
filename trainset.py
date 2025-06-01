

import numpy as np

class Trainset:
    """A trainset contains all useful data for a train model.
    """

    def __init__(
        self,
        ur,
        ir,
        n_users,
        n_items,
        n_ratings,
        rating_scale,
        raw2inner_id_users,
        raw2inner_id_items,
    ):

        self.ur = ur
        self.ir = ir
        self.n_users = n_users
        self.n_items = n_items
        self.n_ratings = n_ratings
        self.rating_scale = rating_scale
        self._raw2inner_id_users = raw2inner_id_users
        self._raw2inner_id_items = raw2inner_id_items
        self._global_mean = None
        # inner2raw dicts could be built right now (or even before) but they
        # are not always useful so we wait until we need them.
        self._inner2raw_id_users = None
        self._inner2raw_id_items = None

    def knows_user(self, uid):
        """Indicate if the user is part of the trainset.
        A user is part of the trainset if the user has at least one rating.
        """

        return uid in self.ur

    def knows_item(self, iid):
        """Indicate if the item is part of the trainset.
        An item is part of the trainset if the item was rated at least once.
        """

        return iid in self.ir

    def to_inner_uid(self, ruid):
        """Convert a **user** raw id to an inner id.
        """

        try:
            return self._raw2inner_id_users[ruid]
        except KeyError:
            raise ValueError("User " + str(ruid) + " is not part of the trainset.")

    def to_raw_uid(self, iuid):
        """Convert a **user** inner id to a raw id.
        """

        if self._inner2raw_id_users is None:
            self._inner2raw_id_users = {
                inner: raw for (raw, inner) in self._raw2inner_id_users.items()
            }

        try:
            return self._inner2raw_id_users[iuid]
        except KeyError:
            raise ValueError(str(iuid) + " is not a valid inner id.")

    def to_inner_iid(self, riid):
        """Convert an **item** raw id to an inner id.
        """

        try:
            return self._raw2inner_id_items[riid]
        except KeyError:
            raise ValueError("Item " + str(riid) + " is not part of the trainset.")

    def to_raw_iid(self, iiid):
        """Convert an **item** inner id to a raw id.
        """

        if self._inner2raw_id_items is None:
            self._inner2raw_id_items = {
                inner: raw for (raw, inner) in self._raw2inner_id_items.items()
            }

        try:
            return self._inner2raw_id_items[iiid]
        except KeyError:
            raise ValueError(str(iiid) + " is not a valid inner id.")

    def all_ratings(self):
        """Generator function to iterate over all ratings.
        """

        for u, u_ratings in self.ur.items():
            for i, r in u_ratings:
                yield u, i, r

    def build_testset(self):
        """Return a list of ratings that can be used as a testset in the
        """

        return [
            (self.to_raw_uid(u), self.to_raw_iid(i), r)
            for (u, i, r) in self.all_ratings()
        ]

    def build_anti_testset(self, fill=None):
        """Return a list of ratings that can be used as a testset in the test()
        method.
        """
        fill = self.global_mean if fill is None else float(fill)

        anti_testset = []
        for u in self.all_users():
            user_items = {j for (j, _) in self.ur[u]}
            anti_testset += [
                (self.to_raw_uid(u), self.to_raw_iid(i), fill)
                for i in self.all_items()
                if i not in user_items
            ]
        return anti_testset

    def all_users(self):
        """Generator function to iterate over all users.
        """
        return range(self.n_users)

    def all_items(self):
        """Generator function to iterate over all items.
        """
        return range(self.n_items)

    @property
    def global_mean(self):
        if self._global_mean is None:
            self._global_mean = np.mean([r for (_, _, r) in self.all_ratings()])

        return self._global_mean
