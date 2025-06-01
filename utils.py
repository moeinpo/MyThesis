
import pandas as pd 
import pickle
import os
import matplotlib.pyplot as plt
import numbers
import numpy as np

def save_matrix_as_pickle(matrix,save_path,filename):
    with open(os.path.join(save_path,filename+'.pkl'), 'wb') as f:
        pickle.dump(matrix, f)


def load_pickle_file(file_path):
    # file_path = 'data_matrix.pkl'
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

def save_matrix_as_excel(matrix,save_path,filename):
    df = pd.DataFrame(matrix)
    df.to_excel(os.path.join(save_path,filename+'.xlsx'), index=False, header=False)

def load_excel_file(file_path):
    #file_path = 'your_file.xlsx'
    df = pd.read_excel(file_path)

def save_matrix_array(matrix,save_path,filename):
    path = os.path.join(save_path, filename)
    with open(path, 'wb') as f:  # 'wb' برای نوشتن در فرمت باینری  
        pickle.dump(matrix, f) 

def read_file_excel_result(cwd,result_storage,file_name,n_fold):
    save_path_result_kfold_number = os.path.join(cwd,result_storage,'CrossValidation',file_name,'k('+str(n_fold+1)+')')
    RMSE_folds = np.array(pd.read_excel(save_path_result_kfold_number+'/'+ str(n_fold)+'alfa_RMSE.xlsx', header=None))
    MSE_folds = np.array(pd.read_excel(save_path_result_kfold_number+'/'+ str(n_fold)+'alfa_MSE.xlsx', header=None))
    MAE_folds = np.array(pd.read_excel(save_path_result_kfold_number+'/'+ str(n_fold)+'alfa_MAE.xlsx', header=None))

    precision_folds = np.array(pd.read_excel(save_path_result_kfold_number+'/'+ str(n_fold)+'alfa_precision.xlsx', header=None))
    recall_folds = np.array(pd.read_excel(save_path_result_kfold_number+'/'+ str(n_fold)+'alfa_recall.xlsx', header=None))
    f1_folds = np.array(pd.read_excel(save_path_result_kfold_number+'/'+ str(n_fold)+'alfa_f1.xlsx', header=None))

    return RMSE_folds,MSE_folds,MAE_folds,precision_folds,recall_folds,f1_folds



def compare_plot(x,Y,algorithms,xlabel,ylabel,save_path,filename):
    plt.figure(figsize=(8, 5))

    n = Y.shape[0]
    for i in range(n):
        plt.plot(x, Y[i], marker='o', label=algorithms[i])

    plt.title('Comparison of Algorithm Outputs')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x)  
    plt.legend()  

    plt.savefig(os.path.join(save_path,filename), format='png')
    #plt.show()
    #plt.close()



def get_rng(random_state):
    """Return a 'validated' RNG.

    If random_state is None, use RandomState singleton from numpy.  Else if
    it's an integer, consider it's a seed and initialized an rng with that
    seed. If it's already an rng, return it.
    """
    if random_state is None:
        return np.random.mtrand._rand
    elif isinstance(random_state, (numbers.Integral, np.integer)):
        return np.random.RandomState(random_state)
    if isinstance(random_state, np.random.RandomState):
        return random_state
    raise ValueError(
        "Wrong random state. Expecting None, an int or a numpy "
        "RandomState instance, got a "
        "{}".format(type(random_state))
    )


    


