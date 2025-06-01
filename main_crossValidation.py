
import numpy as np 
import pandas as pd 
import os
from Datasets import Dataset
from Split import train_test_split
from Split import get_cv
from data_readers import MovieLens_reader,FilmTrust_reader
from Algs import SVD,NMF,ULV,ULV_PSO
import accuracy
from utils import save_matrix_as_excel,compare_plot

data_storage = 'Datasets'
result_storage = 'Results'
#data_filenames = ['MovieLens_100k','filmtrust']
data_filenames = ['MovieLens_100k']

#algorithms = ['SVD','NMF','ULV','ULVPSO']
algorithms = ['SVD','NMF']

for file_name in data_filenames:

    # get current working directory
    cwd = os.getcwd()
    data_path = os.path.join(cwd,data_storage)

    if file_name == 'MovieLens_100k':
        reader = MovieLens_reader
        fullfile_name = os.path.join(data_path,file_name+'.txt')

    if file_name == 'filmtrust':
        reader = FilmTrust_reader
        fullfile_name = os.path.join(data_path,file_name+'.txt')

    # load dataset
    data = Dataset.load_from_file(fullfile_name,reader)

    # split dataset to test and train
    cv = 5
    cv = get_cv(cv)
    cv_num = 0
    for (trainset, testset) in cv.split(data):
        cv_num = cv_num + 1

        # results save path
        save_path = os.path.join(cwd,result_storage,'CrossValidation','cv'+str(cv_num),file_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)


        # learning rates
        #learning_rates = [0.002,0.004,0.006,0.01]
        learning_rates =  np.linspace(0.002,0.01,5)

        m = len(algorithms) 
        n = np.shape(learning_rates)[0]

        n_epochs = 20

        RMSE = np.zeros((m,n))
        MSE = np.zeros((m,n))
        MAE = np.zeros((m,n))
        precision = np.zeros((m,n))
        recall = np.zeros((m,n))
        f1 = np.zeros((m,n))

        for i in range(m):
            algorithm = algorithms[i]

            for j in range(n):
                alfa = learning_rates[j]

                if algorithm == 'SVD':
                    alg = SVD(n_epochs=n_epochs,n_factors=10,lr_all=alfa,biased=True,random_state=1,verbose=True)

                if algorithm == 'NMF':
                    alg = NMF(n_epochs=n_epochs,n_factors=10,lr_bu=alfa, lr_bi=alfa,verbose=True)

                if algorithm == 'ULV':
                    alg = ULV(n_epochs=n_epochs,n_factors=10,lr_all=alfa,verbose=True)

                if algorithm == 'ULVPSO':
                    alg = ULV_PSO(n_epochs=n_epochs,n_factors=10,lr_all=alfa,verbose=True)

                alg.fit(trainset)
                predictions = alg.test(testset)
                RMSE[i,j] = accuracy.rmse(predictions)
                MSE[i,j] = accuracy.mse(predictions)
                MAE[i,j] = accuracy.mae(predictions)
                precision[i,j], recall[i,j], f1[i,j] = accuracy.another_measures(predictions)


        # save results
        save_matrix_as_excel(RMSE,save_path,'alfa_RMSE')  
        save_matrix_as_excel(MSE,save_path,'alfa_MSE')  
        save_matrix_as_excel(MAE,save_path,'alfa_MAE')  
        save_matrix_as_excel(precision,save_path,'alfa_precision')  
        save_matrix_as_excel(recall,save_path,'alfa_recall')  
        save_matrix_as_excel(f1,save_path,'alfa_f1')  
        save_matrix_as_excel(algorithms,save_path,'algorithms')  
        save_matrix_as_excel(learning_rates,save_path,'learning_rates')  

        # plot comparisions
        compare_plot(learning_rates,RMSE,algorithms,'alfa','RMSE',save_path,'alfa_RMSE_cmp.png')
        compare_plot(learning_rates,MSE,algorithms,'alfa','MSE',save_path,'alfa_MSE_cmp.png')
        compare_plot(learning_rates,MAE,algorithms,'alfa','MAE',save_path,'alfa_MAE_cmp.png')
        compare_plot(learning_rates,MAE,algorithms,'alfa','precision',save_path,'alfa_precision_cmp.png')
        compare_plot(learning_rates,MAE,algorithms,'alfa','recall',save_path,'alfa_recall_cmp.png')
        compare_plot(learning_rates,MAE,algorithms,'alfa','f1',save_path,'alfa_f1_cmp.png')

        print('plots saved')