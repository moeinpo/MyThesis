from AlgsBase import AlgsBase
import numpy as np
from predictions import PredictionImpossible
from initialize_pu_qi import init_pu_qi_by_SVD
from initialize_pu_qi import init_pu_pi_by_TULVD
from initialize_pu_qi import init_pu_qi_by_NMF

class SVD(AlgsBase):
    def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 random_state=None, verbose=False):
        
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose

        AlgsBase.__init__(self)

    def fit(self, trainset):
        AlgsBase.fit(self, trainset)
        self.sgd(trainset)
        return self
    
    def sgd(self, trainset):
        rng = np.random.RandomState(self.random_state)
        bu = np.zeros(trainset.n_users, dtype=np.double)
        bi = np.zeros(trainset.n_items, dtype=np.double)

        use_mat_decomposition = True
        if use_mat_decomposition:
            pu, qi = init_pu_qi_by_SVD(trainset, self.n_factors)
        else:
            pu = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_users, self.n_factors))
            qi = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_items, self.n_factors))

        n_factors = self.n_factors
        biased = self.biased
        global_mean = self.trainset.global_mean

        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        lr_pu = self.lr_pu
        lr_qi = self.lr_qi

        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi

        if not biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            se = 0 
            counter = 0
            for u, i, r in trainset.all_ratings():
                dot = 0
                for f in range(n_factors):
                    dot += qi[i, f] * pu[u, f]
                err = r - (global_mean + bu[u] + bi[i] + dot)

                counter = counter + 1
                se = se + err**2

                if biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                for f in range(n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * puf - reg_qi * qif)

            mse = se/counter
            print("mse =  {}".format(mse))

        self.bu = np.asarray(bu)
        self.bi = np.asarray(bi)
        self.pu = np.asarray(pu)
        self.qi = np.asarray(qi)

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean
            if known_user:
                est += self.bu[u]
            if known_item:
                est += self.bi[i]
            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])
        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown.')
        return est

class NMF(AlgsBase):
    def __init__(self, n_factors=15, n_epochs=50, biased=True, reg_pu=.02,
                 reg_qi=.02, reg_bu=.02, reg_bi=.02, lr_bu=.005, lr_bi=.005,
                 init_low=0, init_high=1, random_state=None, verbose=False):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.lr_bu = lr_bu
        self.lr_bi = lr_bi
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.init_low = init_low
        self.init_high = init_high
        self.random_state = random_state
        self.verbose = verbose

        if self.init_low < 0:
            raise ValueError('init_low should be greater than zero')

        AlgsBase.__init__(self)

    def fit(self, trainset):
        AlgsBase.fit(self, trainset)
        self.sgd(trainset)
        return self

    def sgd(self, trainset):
        rng = np.random.RandomState(self.random_state)

        use_mat_decomposition = False
        if use_mat_decomposition:
            pu, qi = init_pu_qi_by_NMF(trainset, self.n_factors)
        else:
            pu = rng.uniform(self.init_low, self.init_high, size=(trainset.n_users, self.n_factors))
            qi = rng.uniform(self.init_low, self.init_high, size=(trainset.n_items, self.n_factors))

        bu = np.zeros(trainset.n_users, dtype=np.double)
        bi = np.zeros(trainset.n_items, dtype=np.double)

        n_factors = self.n_factors
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi
        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        global_mean = self.trainset.global_mean

        user_num = np.zeros((trainset.n_users, n_factors))
        user_denom = np.zeros((trainset.n_users, n_factors))
        item_num = np.zeros((trainset.n_items, n_factors))
        item_denom = np.zeros((trainset.n_items, n_factors))

        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            se = 0 
            counter = 0
            user_num[:, :] = 0
            user_denom[:, :] = 0
            item_num[:, :] = 0
            item_denom[:, :] = 0

            for u, i, r in trainset.all_ratings():
                dot = 0
                for f in range(n_factors):
                    dot += qi[i, f] * pu[u, f]
                est = global_mean + bu[u] + bi[i] + dot
                err = r - est

                counter = counter + 1
                se = se + err**2

                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                for f in range(n_factors):
                    user_num[u, f] += qi[i, f] * r
                    user_denom[u, f] += qi[i, f] * est
                    item_num[i, f] += pu[u, f] * r
                    item_denom[i, f] += pu[u, f] * est

            for u in trainset.all_users():
                n_ratings = len(trainset.ur[u])
                for f in range(n_factors):
                    if pu[u, f] != 0:
                        user_denom[u, f] += n_ratings * reg_pu * pu[u, f]
                        pu[u, f] *= user_num[u, f] / user_denom[u, f]

            for i in trainset.all_items():
                n_ratings = len(trainset.ir[i])
                for f in range(n_factors):
                    if qi[i, f] != 0:
                        item_denom[i, f] += n_ratings * reg_qi * qi[i, f]
                        qi[i, f] *= item_num[i, f] / item_denom[i, f]
            
            mse = se/counter
            print("mse =  {}".format(mse))

        self.bu = np.asarray(bu)
        self.bi = np.asarray(bi)
        self.pu = np.asarray(pu)
        self.qi = np.asarray(qi)

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean
            if known_user:
                est += self.bu[u]
            if known_item:
                est += self.bi[i]
            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])
        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown.')
        return est

class ULV(AlgsBase):
    def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 random_state=None, verbose=False):
        
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose
        AlgsBase.__init__(self)

    def fit(self, trainset):
        AlgsBase.fit(self, trainset)
        self.sgd(trainset)
        return self
    
    def sgd(self, trainset):
        rng = np.random.RandomState(self.random_state)
        bu = np.zeros(trainset.n_users, dtype=np.double)
        bi = np.zeros(trainset.n_items, dtype=np.double)

        use_mat_decomposition = True
        if use_mat_decomposition:
            pu, qi = init_pu_pi_by_TULVD(trainset, self.n_factors)
        else:
            pu = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_users, self.n_factors))
            qi = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_items, self.n_factors))

        n_factors = self.n_factors
        biased = self.biased
        global_mean = self.trainset.global_mean

        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        lr_pu = self.lr_pu
        lr_qi = self.lr_qi

        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi

        if not biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            se = 0 
            counter = 0
            for u, i, r in trainset.all_ratings():
                # dot = 0
                # for f in range(n_factors):
                #     dot += qi[i, f] * pu[u, f]
                dot = np.sum(pu[u] * qi[i])
                err = r - (global_mean + bu[u] + bi[i] + dot)

                counter = counter + 1
                se = se + err**2

                if biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                for f in range(n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * puf - reg_qi * qif)

            mse = se/counter
            print("mse =  {}".format(mse))

        self.bu = np.asarray(bu)
        self.bi = np.asarray(bi)
        self.pu = np.asarray(pu)
        self.qi = np.asarray(qi)

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean
            if known_user:
                est += self.bu[u]
            if known_item:
                est += self.bi[i]
            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])
        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown.')
        return est

class ULV_PSO(AlgsBase):
    def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 random_state=None, verbose=False):
        
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose
        AlgsBase.__init__(self)

    def fit(self, trainset):
        AlgsBase.fit(self, trainset)
        self.sgd(trainset)
        return self
    
    def sgd(self, trainset):
        rng = np.random.RandomState(self.random_state)
        bu = np.zeros(trainset.n_users, dtype=np.double)
        bi = np.zeros(trainset.n_items, dtype=np.double)

        use_mat_decomposition = True
        if use_mat_decomposition:
            pu, qi = init_pu_pi_by_TULVD(trainset, self.n_factors)
        else:
            pu = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_users, self.n_factors))
            qi = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_items, self.n_factors))

        pu_vel = np.zeros((trainset.n_users, self.n_factors), dtype=np.double)
        qi_vel = np.zeros((trainset.n_items, self.n_factors), dtype=np.double)
        pu_personal_best = pu
        qi_personal_best = qi     

        phi1 = 2.05
        phi2 = 2.05
        phi = phi1 + phi2
        chi = 2 / (phi - 2 + np.sqrt(phi**2 - 4 * phi))
        w = chi
        c1 = chi * phi1

        n_factors = self.n_factors
        biased = self.biased
        global_mean = self.trainset.global_mean

        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        lr_pu = self.lr_pu
        lr_qi = self.lr_qi

        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi

        if not biased:
            global_mean = 0

        errs_mat = np.zeros((trainset.n_users, trainset.n_items), dtype=np.double) 
        for u, i, r in trainset.all_ratings():
            dot = 0
            for f in range(n_factors):
                dot += qi[i, f] * pu[u, f]
            err = r - (global_mean + bu[u] + bi[i] + dot)
            errs_mat[u, i] = err

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            se = 0 
            counter = 0
            for u, i, r in trainset.all_ratings():
                pu_vel[u, :] = w * pu_vel[u, :] + c1 * np.random.rand(n_factors) * (pu_personal_best[u, :] - pu[u, :])
                qi_vel[i, :] = w * qi_vel[i, :] + c1 * np.random.rand(n_factors) * (qi_personal_best[i, :] - qi[i, :])

                dot = 0
                for f in range(n_factors):
                    dot += qi[i, f] * pu[u, f]
                err = r - (global_mean + bu[u] + bi[i] + dot)

                counter = counter + 1
                se = se + err**2

                if biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                for f in range(n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * puf - reg_qi * qif)

                if np.random.rand(1)[0] < 0.1:
                    pu[u, :] = pu_vel[u, :]
                    qi[i, :] = qi_vel[i, :]

                if errs_mat[u, i] < err:
                    pu_personal_best[u, :] = pu[u, :]
                    qi_personal_best[i, :] = qi[i, :]

            mse = se / counter
            print("mse =  {}".format(mse))

        self.bu = np.asarray(bu)
        self.bi = np.asarray(bi)
        self.pu = np.asarray(pu)
        self.qi = np.asarray(qi)

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean
            if known_user:
                est += self.bu[u]
            if known_item:
                est += self.bi[i]
            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])
        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown.')
        return est

class ULV_Momentum(AlgsBase):
    def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.005, reg_all=.02,
                 lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 momentum=0.9, random_state=None, verbose=False):
        
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.momentum = momentum
        self.random_state = random_state
        self.verbose = verbose
        AlgsBase.__init__(self)

    def fit(self, trainset):
        AlgsBase.fit(self, trainset)
        self.sgd(trainset)
        return self
    
    def sgd(self, trainset):
        rng = np.random.RandomState(self.random_state)
        bu = np.zeros(trainset.n_users, dtype=np.double)
        bi = np.zeros(trainset.n_items, dtype=np.double)

        use_mat_decomposition = False
        if use_mat_decomposition:
            pu, qi = init_pu_pi_by_TULVD(trainset, self.n_factors)
        else:
            pu = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_users, self.n_factors))
            qi = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_items, self.n_factors))

        # Momentum velocities
        v_bu = np.zeros(trainset.n_users, dtype=np.double)
        v_bi = np.zeros(trainset.n_items, dtype=np.double)
        v_pu = np.zeros((trainset.n_users, self.n_factors), dtype=np.double)
        v_qi = np.zeros((trainset.n_items, self.n_factors), dtype=np.double)

        n_factors = self.n_factors
        biased = self.biased
        global_mean = self.trainset.global_mean

        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        lr_pu = self.lr_pu
        lr_qi = self.lr_qi
        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi
        momentum = self.momentum

        if not biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            se = 0 
            counter = 0
            for u, i, r in trainset.all_ratings():
                # dot = 0
                # for f in range(n_factors):
                #     dot += qi[i, f] * pu[u, f]
                dot = np.sum(pu[u] * qi[i])
                err = r - (global_mean + bu[u] + bi[i] + dot)

                counter = counter + 1
                se = se + err**2

                if biased:
                    # Compute gradients for biases
                    grad_bu = err - reg_bu * bu[u]
                    grad_bi = err - reg_bi * bi[i]
                    # Update velocities for biases
                    v_bu[u] = momentum * v_bu[u] + (1 - momentum) * grad_bu
                    v_bi[i] = momentum * v_bi[i] + (1 - momentum) * grad_bi
                    # Update biases
                    bu[u] += lr_bu * v_bu[u]
                    bi[i] += lr_bi * v_bi[i]

                # Update factors
                for f in range(n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    grad_pu = err * qif - reg_pu * puf
                    grad_qi = err * puf - reg_qi * qif
                    # Update velocities for factors
                    v_pu[u, f] = momentum * v_pu[u, f] + (1 - momentum) * grad_pu
                    v_qi[i, f] = momentum * v_qi[i, f] + (1 - momentum) * grad_qi
                    # Update factors
                    pu[u, f] += lr_pu * v_pu[u, f]
                    qi[i, f] += lr_qi * v_qi[i, f]

            mse = se / counter
            print("mse =  {}".format(mse))

        self.bu = np.asarray(bu)
        self.bi = np.asarray(bi)
        self.pu = np.asarray(pu)
        self.qi = np.asarray(qi)

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean
            if known_user:
                est += self.bu[u]
            if known_item:
                est += self.bi[i]
            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])
        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown.')
        return est



import numpy as np
from AlgsBase import AlgsBase
from predictions import PredictionImpossible
from initialize_pu_qi import init_pu_pi_by_TULVD

class FastULV(AlgsBase):
    def __init__(self, n_factors=100, n_epochs=20, batch_size=1000, biased=True, init_mean=0,
                 init_std_dev=0.1, lr_all=0.005, reg_all=0.02, lr_bu=None, lr_bi=None,
                 lr_pu=None, lr_qi=None, reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 random_state=None, verbose=False, early_stopping=True, patience=3, min_delta=1e-4):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.batch_size = batch_size  # اندازه دسته برای Mini-batch SGD
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping = early_stopping  # فعال‌سازی توقف زودهنگام
        self.patience = patience  # تعداد Epochهای بدون بهبود
        self.min_delta = min_delta  # حداقل بهبود مورد انتظار در MSE
        AlgsBase.__init__(self)

    def fit(self, trainset):
        AlgsBase.fit(self, trainset)
        self.sgd(trainset)
        return self

    def sgd(self, trainset):
        rng = np.random.RandomState(self.random_state)
        bu = np.zeros(trainset.n_users, dtype=np.double)
        bi = np.zeros(trainset.n_items, dtype=np.double)

        # مقداردهی اولیه ماتریس‌های فاکتورها
        use_mat_decomposition = True
        if use_mat_decomposition:
            pu, qi = init_pu_pi_by_TULVD(trainset, self.n_factors)
        else:
            pu = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_users, self.n_factors))
            qi = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_items, self.n_factors))

        # متغیرهای AdaGrad برای نرخ یادگیری تطبیقی
        cache_bu = np.zeros(trainset.n_users, dtype=np.double)
        cache_bi = np.zeros(trainset.n_items, dtype=np.double)
        cache_pu = np.zeros((trainset.n_users, self.n_factors), dtype=np.double)
        cache_qi = np.zeros((trainset.n_items, self.n_factors), dtype=np.double)
        epsilon = 1e-8  # برای جلوگیری از تقسیم بر صفر

        n_factors = self.n_factors
        biased = self.biased
        global_mean = self.trainset.global_mean if self.biased else 0

        # متغیرهای توقف زودهنگام
        best_mse = float('inf')
        patience_counter = 0

        # جمع‌آوری تمام رتبه‌بندی‌ها
        ratings = np.array(list(trainset.all_ratings()), dtype=np.int32)
        n_ratings = len(ratings)

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print(f"Processing epoch {current_epoch}")

            # مخلوط کردن داده‌ها برای Mini-batch
            rng.shuffle(ratings)
            se = 0
            counter = 0

            # پردازش دسته‌ای
            for start_idx in range(0, n_ratings, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_ratings)
                batch = ratings[start_idx:end_idx]
                users = batch[:, 0]
                items = batch[:, 1]
                ratings_batch = batch[:, 2]

                # محاسبه پیش‌بینی‌ها به صورت برداری
                dots = np.sum(pu[users] * qi[items], axis=1)
                estimates = global_mean + bu[users] + bi[items] + dots
                errors = ratings_batch - estimates

                se += np.sum(errors ** 2)
                counter += len(errors)

                # محاسبه گرادیان‌ها
                if biased:
                    grad_bu = errors - self.reg_bu * bu[users]
                    grad_bi = errors - self.reg_bi * bi[items]
                    cache_bu[users] += grad_bu ** 2
                    cache_bi[items] += grad_bi ** 2
                    bu[users] += self.lr_bu * grad_bu / (np.sqrt(cache_bu[users]) + epsilon)
                    bi[items] += self.lr_bi * grad_bi / (np.sqrt(cache_bi[items]) + epsilon)

                # به‌روزرسانی فاکتورها
                for f in range(n_factors):
                    grad_pu = errors * qi[items, f] - self.reg_pu * pu[users, f]
                    grad_qi = errors * pu[users, f] - self.reg_qi * qi[items, f]
                    cache_pu[users, f] += grad_pu ** 2
                    cache_qi[items, f] += grad_qi ** 2
                    pu[users, f] += self.lr_pu * grad_pu / (np.sqrt(cache_pu[users, f]) + epsilon)
                    qi[items, f] += self.lr_qi * grad_qi / (np.sqrt(cache_qi[items, f]) + epsilon)

            mse = se / counter
            print(f"mse = {mse}")

            # بررسی توقف زودهنگام
            if self.early_stopping:
                if mse < best_mse - self.min_delta:
                    best_mse = mse
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping triggered after {current_epoch + 1} epochs")
                        break

        self.bu = np.asarray(bu)
        self.bi = np.asarray(bi)
        self.pu = np.asarray(pu)
        self.qi = np.asarray(qi)

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean
            if known_user:
                est += self.bu[u]
            if known_item:
                est += self.bi[i]
            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])
        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown.')
        return est


# import numpy as np
# import threading
# from queue import Queue
# import random
# import time  # اضافه کردن ماژول time
# from AlgsBase import AlgsBase
# from initialize_pu_qi import init_pu_pi_by_TULVD
# from predictions import PredictionImpossible

# class ULV_DASGA(AlgsBase):
#     def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
#                  init_std_dev=.1, lr_all=.005, reg_all=.02, lr_bu=None, lr_bi=None,
#                  lr_pu=None, lr_qi=None, reg_bu=None, reg_bi=None, reg_pu=None,
#                  reg_qi=None, random_state=None, verbose=False, num_workers=4):
#         self.n_factors = n_factors
#         self.n_epochs = n_epochs
#         self.biased = biased
#         self.init_mean = init_mean
#         self.init_std_dev = init_std_dev
#         self.lr_bu = lr_bu if lr_bu is not None else lr_all
#         self.lr_bi = lr_bi if lr_bi is not None else lr_all
#         self.lr_pu = lr_pu if lr_pu is not None else lr_all
#         self.lr_qi = lr_qi if lr_qi is not None else lr_all
#         self.reg_bu = reg_bu if reg_bu is not None else reg_all
#         self.reg_bi = reg_bi if reg_bi is not None else reg_all
#         self.reg_pu = reg_pu if reg_pu is not None else reg_all
#         self.reg_qi = reg_qi if reg_qi is not None else reg_all
#         self.random_state = random_state
#         self.verbose = verbose
#         self.num_workers = num_workers  # تعداد کارگرها
#         AlgsBase.__init__(self)

#     def fit(self, trainset):
#         AlgsBase.fit(self, trainset)
#         self.sgd(trainset)
#         return self

#     def sgd(self, trainset):
#         rng = np.random.RandomState(self.random_state)
#         bu = np.zeros(trainset.n_users, dtype=np.double)
#         bi = np.zeros(trainset.n_items, dtype=np.double)

#         # مقداردهی اولیه ماتریس‌های pu و qi با استفاده از TULVD
#         use_mat_decomposition = True
#         if use_mat_decomposition:
#             pu, qi = init_pu_pi_by_TULVD(trainset, self.n_factors)
#         else:
#             pu = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_users, self.n_factors))
#             qi = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_items, self.n_factors))

#         # سرور پارامتر برای مدیریت وزن‌ها و گرادیان‌ها
#         param_server = ParameterServer(bu, bi, pu, qi, self.lr_bu, self.lr_bi, self.lr_pu, self.lr_qi,
#                                        self.reg_bu, self.reg_bi, self.reg_pu, self.reg_qi, self.biased)

#         # تقسیم داده‌ها بین کارگرها
#         ratings = list(trainset.all_ratings())
#         np.random.shuffle(ratings)
#         chunk_size = len(ratings) // self.num_workers + 1
#         data_chunks = [ratings[i:i + chunk_size] for i in range(0, len(ratings), chunk_size)]

#         # ایجاد و اجرای کارگرها
#         workers = [
#             Worker(worker_id=i, param_server=param_server, ratings=chunk, global_mean=trainset.global_mean,
#                    n_factors=self.n_factors, verbose=self.verbose)
#             for i, chunk in enumerate(data_chunks[:self.num_workers])
#         ]

#         for current_epoch in range(self.n_epochs):
#             if self.verbose:
#                 print("Processing epoch {}".format(current_epoch))

#             # شروع کارگرها
#             for worker in workers:
#                 worker.set_epoch(current_epoch)
#                 worker.start()

#             # انتظار برای اتمام کارگرها در هر epoch
#             for worker in workers:
#                 worker.join()

#             # محاسبه MSE
#             se = 0
#             counter = 0
#             for u, i, r in trainset.all_ratings():
#                 est = param_server.estimate(u, i, trainset.global_mean)
#                 err = r - est
#                 se += err ** 2
#                 counter += 1
#             mse = se / counter
#             print("mse = {}".format(mse))

#         # ذخیره پارامترهای نهایی
#         self.bu = param_server.bu.copy()
#         self.bi = param_server.bi.copy()
#         self.pu = param_server.pu.copy()
#         self.qi = param_server.qi.copy()

#     def estimate(self, u, i):
#         known_user = self.trainset.knows_user(u)
#         known_item = self.trainset.knows_item(i)

#         if self.biased:
#             est = self.trainset.global_mean
#             if known_user:
#                 est += self.bu[u]
#             if known_item:
#                 est += self.bi[i]
#             if known_user and known_item:
#                 est += np.dot(self.qi[i], self.pu[u])
#         else:
#             if known_user and known_item:
#                 est = np.dot(self.qi[i], self.pu[u])
#             else:
#                 raise PredictionImpossible('User and item are unknown.')
#         return est

# class ParameterServer:
#     def __init__(self, bu, bi, pu, qi, lr_bu, lr_bi, lr_pu, lr_qi, reg_bu, reg_bi, reg_pu, reg_qi, biased):
#         self.bu = bu
#         self.bi = bi
#         self.pu = pu
#         self.qi = qi
#         self.lr_bu = lr_bu
#         self.lr_bi = lr_bi
#         self.lr_pu = lr_pu
#         self.lr_qi = lr_qi
#         self.reg_bu = reg_bu
#         self.reg_bi = reg_bi
#         self.reg_pu = reg_pu
#         self.reg_qi = reg_qi
#         self.biased = biased
#         self.lock = threading.Lock()
#         self.update_queue = Queue()

#     def update_weights(self):
#         while not self.update_queue.empty():
#             grad_bu, grad_bi, grad_pu, grad_qi, u, i, f = self.update_queue.get()
#             with self.lock:
#                 if self.biased:
#                     self.bu[u] += self.lr_bu * (grad_bu - self.reg_bu * self.bu[u])
#                     self.bi[i] += self.lr_bi * (grad_bi - self.reg_bi * self.bi[i])
#                 if f is not None:
#                     self.pu[u, f] += self.lr_pu * (grad_pu - self.reg_pu * self.pu[u, f])
#                     self.qi[i, f] += self.lr_qi * (grad_qi - self.reg_qi * self.qi[i, f])

#     def get_weights(self):
#         with self.lock:
#             return self.bu.copy(), self.bi.copy(), self.pu.copy(), self.qi.copy()

#     def estimate(self, u, i, global_mean):
#         bu, bi, pu, qi = self.get_weights()
#         est = global_mean if self.biased else 0
#         est += bu[u] if self.biased else 0
#         est += bi[i] if self.biased else 0
#         est += np.dot(qi[i], pu[u])
#         return est

# class Worker(threading.Thread):
#     def __init__(self, worker_id, param_server, ratings, global_mean, n_factors, verbose=False):
#         super().__init__()
#         self.worker_id = worker_id
#         self.param_server = param_server
#         self.ratings = ratings
#         self.global_mean = global_mean
#         self.n_factors = n_factors
#         self.verbose = verbose
#         self.current_epoch = 0

#     def set_epoch(self, epoch):
#         self.current_epoch = epoch

#     def run(self):
#         for u, i, r in self.ratings:
#             # دریافت وزن‌های فعلی
#             bu, bi, pu, qi = self.param_server.get_weights()

#             # محاسبه خطا
#             dot = 0
#             for f in range(self.n_factors):
#                 dot += qi[i, f] * pu[u, f]
#             err = r - (self.global_mean + bu[u] + bi[i] + dot)

#             # محاسبه گرادیان‌ها
#             if self.param_server.biased:
#                 grad_bu = err
#                 grad_bi = err
#                 self.param_server.update_queue.put((grad_bu, grad_bi, None, None, u, i, None))

#             for f in range(self.n_factors):
#                 grad_pu = err * qi[i, f]
#                 grad_qi = err * pu[u, f]
#                 self.param_server.update_queue.put((None, None, grad_pu, grad_qi, u, i, f))

#             # شبیه‌سازی تاخیر ناهمزمان
#             time.sleep(random.uniform(0.01, 0.05))

#             # به‌روزرسانی وزن‌ها
#             self.param_server.update_weights()




# class ULV_DASGA(AlgsBase):
#     def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
#                  init_std_dev=.1, lr_all=.005,
#                  reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
#                  reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
#                  random_state=None, verbose=False):
        
#         self.n_factors = n_factors
#         self.n_epochs = n_epochs
#         self.biased = biased
#         self.init_mean = init_mean
#         self.init_std_dev = init_std_dev
#         self.lr_bu = lr_bu if lr_bu is not None else lr_all
#         self.lr_bi = lr_bi if lr_bi is not None else lr_all
#         self.lr_pu = lr_pu if lr_pu is not None else lr_all
#         self.lr_qi = lr_qi if lr_qi is not None else lr_all
#         self.reg_bu = reg_bu if reg_bu is not None else reg_all
#         self.reg_bi = reg_bi if reg_bi is not None else reg_all
#         self.reg_pu = reg_pu if reg_pu is not None else reg_all
#         self.reg_qi = reg_qi if reg_qi is not None else reg_all
#         self.random_state = random_state
#         self.verbose = verbose
#         AlgsBase.__init__(self)

#     def fit(self, trainset):
#         AlgsBase.fit(self, trainset)
#         self.sgd(trainset)
#         return self
    
#     def sgd(self, trainset):
#         rng = np.random.RandomState(self.random_state)
#         bu = np.zeros(trainset.n_users, dtype=np.double)
#         bi = np.zeros(trainset.n_items, dtype=np.double)

#         use_mat_decomposition = True
#         if use_mat_decomposition:
#             pu, qi = init_pu_pi_by_TULVD(trainset, self.n_factors)
#         else:
#             pu = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_users, self.n_factors))
#             qi = rng.normal(self.init_mean, self.init_std_dev, size=(trainset.n_items, self.n_factors))

#         # Initialize adaptive learning rates
#         lr_pu_adaptive = np.ones((trainset.n_users, self.n_factors)) * self.lr_pu
#         lr_qi_adaptive = np.ones((trainset.n_items, self.n_factors)) * self.lr_qi
#         lr_bu_adaptive = np.ones(trainset.n_users) * self.lr_bu
#         lr_bi_adaptive = np.ones(trainset.n_items) * self.lr_bi

#         # Initialize momentum terms
#         momentum_pu = np.zeros((trainset.n_users, self.n_factors))
#         momentum_qi = np.zeros((trainset.n_items, self.n_factors))
#         momentum_bu = np.zeros(trainset.n_users)
#         momentum_bi = np.zeros(trainset.n_items)

#         n_factors = self.n_factors
#         biased = self.biased
#         global_mean = self.trainset.global_mean

#         reg_bu = self.reg_bu
#         reg_bi = self.reg_bi
#         reg_pu = self.reg_pu
#         reg_qi = self.reg_qi

#         if not biased:
#             global_mean = 0

#         # DASGA parameters
#         beta = 0.9  # Momentum coefficient
#         alpha = 0.001  # Learning rate decay
#         epsilon = 1e-8  # Small constant for numerical stability

#         for current_epoch in range(self.n_epochs):
#             if self.verbose:
#                 print("Processing epoch {}".format(current_epoch))

#             se = 0 
#             counter = 0
#             for u, i, r in trainset.all_ratings():
#                 dot = 0
#                 for f in range(n_factors):
#                     dot += qi[i, f] * pu[u, f]
#                 err = r - (global_mean + bu[u] + bi[i] + dot)

#                 counter = counter + 1
#                 se = se + err**2

#                 if biased:
#                     # Update user and item biases with DASGA
#                     grad_bu = err - reg_bu * bu[u]
#                     grad_bi = err - reg_bi * bi[i]
                    
#                     momentum_bu[u] = beta * momentum_bu[u] + (1 - beta) * grad_bu**2
#                     momentum_bi[i] = beta * momentum_bi[i] + (1 - beta) * grad_bi**2
                    
#                     lr_bu_adaptive[u] = self.lr_bu / (np.sqrt(momentum_bu[u]) + epsilon)
#                     lr_bi_adaptive[i] = self.lr_bi / (np.sqrt(momentum_bi[i]) + epsilon)
                    
#                     bu[u] += lr_bu_adaptive[u] * grad_bu
#                     bi[i] += lr_bi_adaptive[i] * grad_bi

#                 for f in range(n_factors):
#                     puf = pu[u, f]
#                     qif = qi[i, f]
                    
#                     # Compute gradients
#                     grad_pu = err * qif - reg_pu * puf
#                     grad_qi = err * puf - reg_qi * qif
                    
#                     # Update momentum
#                     momentum_pu[u, f] = beta * momentum_pu[u, f] + (1 - beta) * grad_pu**2
#                     momentum_qi[i, f] = beta * momentum_qi[i, f] + (1 - beta) * grad_qi**2
                    
#                     # Update adaptive learning rates
#                     lr_pu_adaptive[u, f] = self.lr_pu / (np.sqrt(momentum_pu[u, f]) + epsilon)
#                     lr_qi_adaptive[i, f] = self.lr_qi / (np.sqrt(momentum_qi[i, f]) + epsilon)
                    
#                     # Update parameters
#                     pu[u, f] += lr_pu_adaptive[u, f] * grad_pu
#                     qi[i, f] += lr_qi_adaptive[i, f] * grad_qi

#             # Decay learning rates
#             self.lr_pu *= (1 - alpha)
#             self.lr_qi *= (1 - alpha)
#             self.lr_bu *= (1 - alpha)
#             self.lr_bi *= (1 - alpha)

#             mse = se/counter
#             print("mse =  {}".format(mse))

#         self.bu = np.asarray(bu)
#         self.bi = np.asarray(bi)
#         self.pu = np.asarray(pu)
#         self.qi = np.asarray(qi)

#     def estimate(self, u, i):
#         known_user = self.trainset.knows_user(u)
#         known_item = self.trainset.knows_item(i)

#         if self.biased:
#             est = self.trainset.global_mean
#             if known_user:
#                 est += self.bu[u]
#             if known_item:
#                 est += self.bi[i]
#             if known_user and known_item:
#                 est += np.dot(self.qi[i], self.pu[u])
#         else:
#             if known_user and known_item:
#                 est = np.dot(self.qi[i], self.pu[u])
#             else:
#                 raise PredictionImpossible('User and item are unknown.')
#         return est

