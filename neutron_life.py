import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from math import factorial


####input data
df_data = pd.read_csv("mean_life.txt" ,delim_whitespace=True, header=0, skiprows = None, usecols = ["Mean_Lifetime", "error1", "error2"])
df_data = df_data.replace({np.nan:0})

######calculating total error and adding it as a column
df_data["total_error"] = np.sqrt((df_data.error1)**2 + (df_data.error2)**2)
df_data = df_data.sort_values(by = ["Mean_Lifetime"])

#####Number of data points
N = 19

########calculating the probability in order to calculate median
prob = [((2**-N)*factorial(N))/(factorial(i)*factorial(N-i)) for i in range(1, N+1) ]

#prob1 = prob[0:23]
#####including probability as a column in dataframe
df_data["prob"] = prob

######probability vs N
plt.plot(np.arange(1,N+1,1),prob)
plt.show()


data_arr = df_data.as_matrix()


####Median 
if N%2 ==0:
    median = (data_arr[int((N-1)/2), 0] + data_arr[int((N-1)/2)+1, 0])/2
#    sum = data_arr[int((N-1)/2), 4]
#    for i in range(0,N):
#       sum = sum + data_arr[int((N-1)/2) - i,4] + data_arr[int((N-1)/2) + i, 4]
#       if sum >= 0.6827 :
#            med_error_l = data_arr[int((N-1)/2) - i,3]
#            med_error_u = data_arr[int((N-1)/2) + i,3]
#            break

else :
    median = data_arr[int((N-1)/2), 0]
#    sum = 0
#    for i in range(1,N):
#        sum = sum + data_arr[int((N-1)/2) - i,4]
#        if sum >= 0.6827 :
#            med_error_l = data_arr[int((N-1)/2) - i,3]
#            med_error_u = data_arr[int((N-1)/2) + (i-1),3]
#            break
#        
#        sum = sum + data_arr[int((N-1)/2)+(i-1), 4]
#        
#        if sum >= 0.6827 :
#            med_error_l = data_arr[int((N-1)/2) - i,3]
#            med_error_u = data_arr[int((N-1)/2) + (i-1),3]
#            break

#######calculating median error
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

#####generating the error distributions and adding them as columns to initial dataframe
df_data["N_wm_p"] = (df_data.Mean_Lifetime - w_mean)/np.sqrt((df_data.total_error)**2 + sigma_wm**2)
df_data["N_wm_n"] = (df_data.Mean_Lifetime - w_mean)/np.sqrt((df_data.total_error)**2 - sigma_wm**2)
df_data["N_am"] = (df_data.Mean_Lifetime - a_mean)/np.sqrt((df_data.total_error)**2 + sigma_sd**2)
df_data["N_med"] = (df_data.Mean_Lifetime - median)/np.sqrt((df_data.total_error)**2 + sigma_med**2)

col=["N_wm_p", "N_wm_n", "N_am", "N_med"]


######Creating an array of error distributions, the columns are "N_wm_p", "N_wm_n", "N_am", "N_med" in this order
data_arr = df_data.as_matrix(columns = col)

######negative of above data array
data_arr_ = np.negative(data_arr) 

#######joining both the arrays, actual array + negatives(Symmetrizing)
data_arr = np.concatenate((data_arr,data_arr_), axis=0)

########################for scale factor = 1######################################



ks_arr= np.zeros((4,4))

###p-values for scal factor =1 for all distributions(weighted mean+), for t distribution we check for dof from 1 to 2000 and find maximum
D_norm, ks_arr[0,0] = stats.kstest(data_arr[:,0], 'norm')
D_norm, ks_arr[0,1] = stats.kstest(data_arr[:,0], 'cauchy')
D_norm, ks_arr[0,2] = stats.kstest(data_arr[:,0], 'laplace')

max_val=0
for i in range(2,2001):
    D_norm, v1 = stats.kstest(data_arr[:,0], 't', args=(i,0,1))
    
    if v1>max_val:
        print(v1)
        max_val = v1
        n=i
ks_arr[0,3] = max_val
print(n)
###p-values forcale factor=1 for all distributions(weighted mean-)
D_norm, ks_arr[1,0] = stats.kstest(data_arr[:,1], 'norm')
D_norm, ks_arr[1,1] = stats.kstest(data_arr[:,1], 'cauchy')
D_norm, ks_arr[1,2] = stats.kstest(data_arr[:,1], 'laplace')
max_val=0
for i in range(2,2001):
    D_norm, v1 = stats.kstest(data_arr[:,1], 't', args=(i,0,1))
    if v1>max_val:
        max_val = v1
        n=i
ks_arr[1,3] = max_val
print(n)

###p-values for cale factor=1 for all distributions(arithmetic mean)
D_norm, ks_arr[2,0] = stats.kstest(data_arr[:,2], 'norm')
D_norm, ks_arr[2,1] = stats.kstest(data_arr[:,2], 'cauchy')
D_norm, ks_arr[2,2] = stats.kstest(data_arr[:,2], 'laplace')
max_val=0
for i in range(2,2001):
    D_norm, v1 = stats.kstest(data_arr[:,2], 't', args=(i,0,1))
    if v1>max_val:
        max_val = v1
        n=i
ks_arr[2,3] = max_val
print(n)


###p-values for scale factor=1 for all distributions(median)
D_norm, ks_arr[3,0] = stats.kstest(data_arr[:,3], 'norm')
D_norm, ks_arr[3,1] = stats.kstest(data_arr[:,3], 'cauchy')
D_norm, ks_arr[3,2] = stats.kstest(data_arr[:,3], 'laplace')
max_val=0
for i in range(2,2001):
    D_norm, v1 = stats.kstest(data_arr[:,3], 't', args=(i,0,1))
    if v1>max_val:
        max_val = v1
        n=i
ks_arr[3,3] = max_val
print(n)


#######KS tests for different scale factors for norm, cauchy and laplace, 
#######first column = scale factor next 3 columns for wm+(norm, cauchy, laplace order), next 3 for wm-, next 3 for median
#data_arr = df_data.as_matrix()

ks_arr_scale= np.zeros((2500,10))

for i, s in enumerate(np.arange(0.001, 2.501, 0.001)):
    
    ks_arr_scale[i,0] = s
    
    ###p-values for different scale factors for all distributions(weighted mean+)
    D_norm, ks_arr_scale[i,1] = stats.kstest(data_arr[:,0], 'norm', args =(0,s))
    D_norm, ks_arr_scale[i,2] = stats.kstest(data_arr[:,0], 'cauchy', args =(0,s))
    D_norm, ks_arr_scale[i,3] = stats.kstest(data_arr[:,0], 'laplace', args =(0,s))
    
    
    ###p-values for different scale factors for all distributions(weighted mean-)
    D_norm, ks_arr_scale[i,4] = stats.kstest(data_arr[:,1], 'norm', args =(0,s))
    D_norm, ks_arr_scale[i,5] = stats.kstest(data_arr[:,1], 'cauchy', args =(0,s))
    D_norm, ks_arr_scale[i,6] = stats.kstest(data_arr[:,1], 'laplace', args =(0,s))
   
    
#    ###p-values for different scale factors for all distributions(arithmetic mean)
#    D_norm, ks_arr_scale[i,7] = stats.kstest(data_arr[:,2], 'norm', args =(0,s))
#    D_norm, ks_arr_scale[i,8] = stats.kstest(data_arr[:,2], 'cauchy', args =(0,s))
#    D_norm, ks_arr_scale[i,9] = stats.kstest(data_arr[:,2], 'laplace', args =(0,s))

    
    ###p-values for different scale factors for all distributions(median)
    D_norm, ks_arr_scale[i,7] = stats.kstest(data_arr[:,3], 'norm', args =(0,s))
    D_norm, ks_arr_scale[i,8] = stats.kstest(data_arr[:,3], 'cauchy', args =(0,s))
    D_norm, ks_arr_scale[i,9] = stats.kstest(data_arr[:,3], 'laplace', args =(0,s))
    

indices = np.argmax(ks_arr_scale, axis=0)





max_arr = np.zeros((9,2))

for i in range(1,10):
    max_arr[i-1,0]= ks_arr_scale[indices[i],0]
    max_arr[i-1,1]= ks_arr_scale[indices[i],i]


    
n=2000


######This part finds 3 arrays for t distribution, each for wm+wm- and median, rows = scale factors, columns = dof (n) starting from n=2, ks_arr_t1 for wm+, ks_arr_t2 for wm-, ks_arr_t3 for median

######This makes the compilation really slow.

ks_arr_t1= np.zeros((250,n+1))
ks_arr_t2= np.zeros((250,n+1))
ks_arr_t3= np.zeros((250,n+1))


for i, s in enumerate(np.arange(0.001, 2.501, 0.01)):
    for n in range(2,2001):
        #ks_arr_t1[i,0] = s
        #ks_arr_t2[i,0] = s
        #ks_arr_t3[i,0] = s
        D_norm, ks_arr_t1[i,n] = stats.kstest(data_arr[:,0], 't', args =(n, 0,s))
        D_norm, ks_arr_t2[i,n] = stats.kstest(data_arr[:,1], 't', args =(n, 0,s))
        D_norm, ks_arr_t3[i,n] = stats.kstest(data_arr[:,3], 't', args =(n, 0,s))

############x1, x2, x3 are the indices of the max values in ks_arr_t1, ks_arr_t2, ks_arr_t3. This way you can also find the scale factor for which max p-value occurs by using 0.001 + (x1.row_number +1)0.01
x1 = np.unravel_index(ks_arr_t1.argmax(), ks_arr_t1.shape)
x2 = np.unravel_index(ks_arr_t2.argmax(), ks_arr_t2.shape)
x3 = np.unravel_index(ks_arr_t3.argmax(), ks_arr_t2.shape)