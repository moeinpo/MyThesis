
import numpy as np 
import pandas as pd 
import os
from Datasets import Dataset
from Split import train_test_split
from data_readers import MovieLens_reader,FilmTrust_reader
from Algs import SVD,NMF,ULV,ULV_PSO
import accuracy
from utils import save_matrix_as_excel,compare_plot

data_storage = 'Datasets'
result_storage = 'Results'
#data_filenames = ['MovieLens_100k','filmtrust']
data_filenames = ['MovieLens_100k','filmtrust']

algorithms = ['SVD','NMF','ULV','ULVPSO']

for file_name in data_filenames:

    # get current working directory
    cwd = os.getcwd()
    data_path = os.path.join(cwd,data_storage)

    save_path = os.path.join(cwd,result_storage,'test_train_split',file_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if file_name == 'MovieLens_100k':
        reader = MovieLens_reader
        fullfile_name = os.path.join(data_path,file_name+'.txt')

    if file_name == 'filmtrust':
        reader = FilmTrust_reader
        fullfile_name = os.path.join(data_path,file_name+'.txt')

    # load dataset
    data = Dataset.load_from_file(fullfile_name,reader)

    # split dataset to test and train
    trainset,testset = train_test_split(data,test_size=0.1)

    # learning rates
    #alfa = [0.002,0.004,0.006,0.01]
    learning_rates =  np.linspace(0.002,0.01,5)

    m = len(algorithms) 
    n = np.shape(learning_rates)[0]

    n_epochs = 20

    RMSE = np.zeros((m,n))
    MSE = np.zeros((m,n))
    MAE = np.zeros((m,n))

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


    # save results
    save_matrix_as_excel(RMSE,save_path,'alfa_RMSE')  
    save_matrix_as_excel(MSE,save_path,'alfa_MSE')  
    save_matrix_as_excel(MAE,save_path,'alfa_MAE')  
    save_matrix_as_excel(algorithms,save_path,'algorithms')  
    save_matrix_as_excel(learning_rates,save_path,'learning_rates')  

    # plot comparisions
    compare_plot(learning_rates,RMSE,algorithms,'alfa','RMSE',save_path,'alfa_RMSE_cmp.png')
    compare_plot(learning_rates,MSE,algorithms,'alfa','MSE',save_path,'alfa_MSE_cmp.png')
    compare_plot(learning_rates,MAE,algorithms,'alfa','MAE',save_path,'alfa_MAE_cmp.png')
    