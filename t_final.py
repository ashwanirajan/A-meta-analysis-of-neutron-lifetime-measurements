import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from math import factorial
import random

####input data
n = 1000
ks_arr= np.zeros((n,3))
ks_arr_scale = np.zeros((n,3))
for j in range(0,n):
    random.seed()
    x=stats.t.rvs(881.5,0.47,2,19)
    df_data = pd.read_csv("mean_life.txt" ,delim_whitespace=True, header=0, skiprows = None, usecols = ["Mean_Lifetime", "error1", "error2"])
    df_data = df_data.replace({np.nan:0})
    
    df_data["total_error"] = np.sqrt((df_data.error1)**2 + (df_data.error2)**2)
    df_data["Mean_Lifetime"] = x
    df_data = df_data.sort_values(by = ["Mean_Lifetime"])
    N = 19
    
    prob = [((2**-N)*factorial(N))/(factorial(i)*factorial(N-i)) for i in range(1, N+1) ]
    #prob1 = prob[0:23]
    df_data["prob"] = prob
    
    plt.plot(np.arange(1,N+1,1),prob)
    
    
    data_arr = df_data.as_matrix()
    ####Median and error
     
    
    if N%2 ==0:
        median = (data_arr[int((N-1)/2), 0] + data_arr[int((N-1)/2)+1, 0])/2
    
    else :
        median = data_arr[int((N-1)/2), 0]
    
    
    C=[np.sum(newx for newx in prob[k:N-k]) for k in range(0,int(N/2)+1)];
    
    for l in range(len(C)):
        if(C[l]<=0.6827):
            break;
    
    
    
    sigma_med = np.abs(data_arr[l,3] - data_arr[N-l,3])/2
    #####Weighted mean and error
    
    w_mean = np.sum(data_arr[:,0]/(data_arr[:,3]**2))/np.sum(1.0/data_arr[:,3]**2)
    
    sigma_wm = 1.0/np.sqrt(np.sum(1.0/data_arr[:,3]**2))
    
    ######Arithmetic mean and error(SD)
    
    a_mean = np.sum(data_arr[:,0])/N
    
    sigma_sd = np.sqrt(np.sum((data_arr[:,0]-a_mean)**2)/N**2)
    
    #####error distributions
    df_data["N_wm_p"] = (df_data.Mean_Lifetime - w_mean)/np.sqrt((df_data.total_error)**2 + sigma_wm**2)
    df_data["N_wm_n"] = (df_data.Mean_Lifetime - w_mean)/np.sqrt((df_data.total_error)**2 - sigma_wm**2)
    #df_data["N_am"] = (df_data.Mean_Lifetime - a_mean)/np.sqrt((df_data.total_error)**2 + sigma_sd**2)
    df_data["N_med"] = (df_data.Mean_Lifetime - median)/np.sqrt((df_data.total_error)**2 + sigma_med**2)
    
    col=["N_wm_p", "N_wm_n", "N_med"]
    
    data_arr = df_data.as_matrix(columns = col)
    data_arr_ = np.negative(data_arr) 
    
    
    data_arr = np.concatenate((data_arr,data_arr_), axis=0)
    
    
    ######for scale =1
    
    
    ###p-values for scal factor =1 for all distributions(weighted mean+)
    D_norm, v1 = stats.kstest(data_arr[:,0], 't', args=(2,0,1))
    ks_arr[j,0] = v1
        #print(n)
    
    ###p-values forcale factor=1 for all distributions(weighted mean-)
    D_norm, v1 = stats.kstest(data_arr[:,1], 't', args=(2,0,1))
    ks_arr[j,1] = v1

    
    D_norm, v1 = stats.kstest(data_arr[:,2], 't', args=(2,0,1))
    ks_arr[j,2] = v1
        #print(n)
        
    
    
    D_norm, ks_arr_scale[j,0] = stats.kstest(data_arr[:,0], 't', args =(2, 0,1.091))
    D_norm, ks_arr_scale[j,1] = stats.kstest(data_arr[:,1], 't', args =(2, 0,1.201))
    #D_norm, ks_arr_scale[i,2] = stats.ks_2samp(binned_am, stats.norm.pdf(x = center_am, scale =s))
    D_norm, ks_arr_scale[j,2] = stats.kstest(data_arr[:,2], 't', args =(2, 0,1.021))

result = np.zeros((6,3))
for k in range(0,3):
    result[k,0] = np.mean(ks_arr[:,k])
    result[k,1] = np.median(ks_arr[:,k])
    result[k,2] = 0.7413 * (stats.iqr(ks_arr[:,k],axis =0))

for k in range(3,6):
    result[k,0] = np.mean(ks_arr_scale[:,k-3])
    result[k,1] = np.median(ks_arr_scale[:,k-3])
    result[k,2] = 0.7413 * (stats.iqr(ks_arr_scale[:,k-3],axis =0))
