

""" initialize user factor(pu) and item factor(qi)
 according to the matrix decomposition technique

 for Eaxaple:
 If the SVD = U.S.V is used, then pu = U, qi = V  
 """
import numpy as np
from ulv_tools import hulv
from sklearn.decomposition import NMF  

def init_pu_qi_by_SVD(trainset=None,n_factors = 100):
    if trainset is None:
        print('trainset is None. please set valid trainset')
        return
    trainset_pivot = np.zeros((trainset.n_users,trainset.n_items),dtype=float)
    for u, i, r in trainset.all_ratings():
        trainset_pivot[u,i]=r

    # Computing the SVD of the input matrix
    U, S, VT = np.linalg.svd(trainset_pivot)
    max_n_factors = np.min([trainset.n_users,trainset.n_items])
    if n_factors>max_n_factors:
        n_factors = max_n_factors
    pu = U[:,:n_factors]
    qi = VT[:n_factors,:].T

    return pu,qi




def init_pu_qi_by_NMF(trainset=None,n_factors = 100):
    if trainset is None:
        print('trainset is None. please set valid trainset')
        return
    trainset_pivot = np.zeros((trainset.n_users,trainset.n_items),dtype=float)
    for u, i, r in trainset.all_ratings():
        trainset_pivot[u,i]=r

    # Computing the SVD of the input matrix
    n_components = n_factors 

# اجرای NMF  
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=1000, tol=1e-5)  
    W = model.fit_transform(trainset_pivot)  # ماتریس ویژگی‌ها  
    H = model.components_  # ماتریس بارگذاری 

    pu = W
    qi = H.T

    return pu,qi
    

def init_pu_pi_by_TULVD(trainset=None,n_factors = 100):
    if trainset is None:
        print('trainset is None. please set valid trainset')
        return
    trainset_pivot = np.zeros((trainset.n_users,trainset.n_items),dtype=float)
    for u, i, r in trainset.all_ratings():
        trainset_pivot[u,i]=r

    # Define input parameters.
    tol_rank = 0.001
    tol_ref  = 1e-04
    max_ref  = 2

    m,n = trainset_pivot.shape
    if m<n:
        A = np.transpose(trainset_pivot)
        # Compute ULV.
        _,L,U,V = hulv(A,tol_rank,tol_ref,max_ref)
        pu = U[:,:n_factors]
        qi = V[:,:n_factors]            
    else:
        A = trainset_pivot    
        
        # Compute ULV.
        _,L,V,U = hulv(A,tol_rank,tol_ref,max_ref)
        pu = U[:,:n_factors]
        qi = V[:,:n_factors]

    return pu,qi

