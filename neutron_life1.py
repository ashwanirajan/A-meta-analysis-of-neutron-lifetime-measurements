import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from math import factorial

####input data

df_data = pd.read_csv("mean_life.txt" ,delim_whitespace=True, header=0, skiprows = None, usecols = ["Mean_Lifetime", "error1", "error2"])
df_data = df_data.replace({np.nan:0})

df_data["total_error"] = np.sqrt((df_data.error1)**2 + (df_data.error2)**2)
df_data = df_data.sort_values(by = ["Mean_Lifetime"])
N = 19

prob = [((2**-N)*factorial(N))/(factorial(i)*factorial(N-i)) for i in range(1, N+1) ]
#prob1 = prob[0:23]
df_data["prob"] = prob

plt.plot(np.arange(1,N+1,1),prob)
plt.show()

data_arr = df_data.as_matrix()
####Median and error
 

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
df_data["N_am"] = (df_data.Mean_Lifetime - a_mean)/np.sqrt((df_data.total_error)**2 + sigma_sd**2)
df_data["N_med"] = (df_data.Mean_Lifetime - median)/np.sqrt((df_data.total_error)**2 + sigma_med**2)

col=["N_wm_p", "N_wm_n", "N_am", "N_med"]

data_arr = df_data.as_matrix(columns = col)
data_arr_ = np.negative(data_arr) 


data_arr = np.concatenate((data_arr,data_arr_), axis=0)


################For wmp, generating freq.distribution and pdf values at bin center
bin_size = 0.5
vmax=round(np.max(abs(data_arr[:,0])),1)+0.1 # find the maximum absolute value upto 1st decimal place
bins_wmp=np.arange(-vmax,vmax,bin_size)
binned_wmp,b, c = plt.hist(data_arr[:,0],bins=bins_wmp, normed =True)
center_wmp = 0.5*(bins_wmp[:-1] + bins_wmp[1:]) 
z_wmp_g=stats.norm.pdf(center_wmp)
z_wmp_c=stats.cauchy.pdf(center_wmp)
z_wmp_l=stats.laplace.pdf(center_wmp)
z_wmp_t=stats.t.pdf(center_wmp, 1)


################For wmp, generating freq.distribution and pdf values at bin center
vmax=round(np.max(abs(data_arr[:,1])),1)+0.1 # find the maximum absolute value upto 1st decimal place
bins_wmn=np.arange(-vmax,vmax,bin_size)
binned_wmn,b, c = plt.hist(data_arr[:,1],bins=bins_wmn, normed =True)
center_wmn = 0.5*(bins_wmn[:-1] + bins_wmn[1:]) 
z_wmn_g=stats.norm.pdf(center_wmn)
z_wmn_c=stats.cauchy.pdf(center_wmn)
z_wmn_l=stats.laplace.pdf(center_wmn)
z_wmn_t=stats.t.pdf(center_wmn,1)

################For am, generating freq.distribution and pdf values at bin center
vmax=round(np.max(abs(data_arr[:,2])),1)+0.1 # find the maximum absolute value upto 1st decimal place
bins_am=np.arange(-vmax,vmax,bin_size)
binned_am,b, c = plt.hist(data_arr[:,2],bins=bins_am, normed =True)
center_am = 0.5*(bins_am[:-1] + bins_am[1:]) 
z_am_g=stats.norm.pdf(center_am)
z_am_c=stats.cauchy.pdf(center_am)
z_am_l=stats.laplace.pdf(center_am)
z_am_t=stats.t.pdf(center_am,1)


################For median, generating freq.distribution and pdf values at bin center
vmax=round(np.max(abs(data_arr[:,3])),1)+0.1 # find the maximum absolute value upto 1st decimal place
bins_med=np.arange(-vmax,vmax,bin_size)
binned_med,b, c = plt.hist(data_arr[:,3],bins=bins_med, normed =True)
center_med = 0.5*(bins_med[:-1] + bins_med[1:]) 
z_med_g=stats.norm.pdf(center_med)
z_med_c=stats.cauchy.pdf(center_med)
z_med_l=stats.laplace.pdf(center_med)
z_med_t=stats.t.pdf(center_med,1)


######for scale =1
ks_arr= np.zeros((4,4))















##################Test#########################################


#alp1, bet1 = stats.ks_2samp( stats.cauchy.pdf(center_wmp),binned_wmp)
#alp2, bet2 = stats.ks_2samp(stats.norm.pdf(center_wmp),binned_wmp)
#alp3, bet3 = stats.ks_2samp(stats.norm.pdf(center_wmp),stats.cauchy.pdf(center_wmp))
#alp4, bet4 = stats.ks_2samp(stats.norm.pdf(center_wmp),stats.laplace.pdf(center_wmp))








###p-values for scal factor =1 for all distributions(weighted mean+)
D_norm, ks_arr[0,0] = stats.ks_2samp(binned_wmp, z_wmp_g)
D_norm, ks_arr[0,1] = stats.ks_2samp(binned_wmp, z_wmp_c)
D_norm, ks_arr[0,2] = stats.ks_2samp(binned_wmp, z_wmp_l)

max_val=0
for i in range(2,2001):
    D_norm, v1 = stats.ks_2samp(binned_wmp, stats.t.pdf(center_wmp, df = i))
    
    if v1>max_val:
        print(v1)
        max_val = v1
        n=i
ks_arr[0,3] = max_val
print(n)
###p-values forcale factor=1 for all distributions(weighted mean-)
D_norm, ks_arr[1,0] = stats.ks_2samp(binned_wmn, z_wmn_g)
D_norm, ks_arr[1,1] = stats.ks_2samp(binned_wmn, z_wmn_c)
D_norm, ks_arr[1,2] = stats.ks_2samp(binned_wmn, z_wmn_l)
max_val=0
for i in range(2,2001):
    D_norm, v1 = stats.ks_2samp(binned_wmn, stats.t.pdf(center_wmn, df = i))
    if v1>max_val:
        max_val = v1
        n=i
ks_arr[1,3] = max_val
print(n)

###p-values for cale factor=1 for all distributions(arithmetic mean)
D_norm, ks_arr[2,0] = stats.ks_2samp(binned_am, z_am_g)
D_norm, ks_arr[2,1] = stats.ks_2samp(binned_am, z_am_c)
D_norm, ks_arr[2,2] = stats.ks_2samp(binned_am, z_am_l)
max_val=0
for i in range(2,2001):
    D_norm, v1 = stats.ks_2samp(binned_am, stats.t.pdf(center_am, df = i))
    if v1>max_val:
        max_val = v1
        n=i
ks_arr[2,3] = max_val
print(n)


###p-values for scale factor=1 for all distributions(median)
D_norm, ks_arr[3,0] = stats.ks_2samp(binned_med, z_med_g)
D_norm, ks_arr[3,1] = stats.ks_2samp(binned_med, z_med_c)
D_norm, ks_arr[3,2] = stats.ks_2samp(binned_med, z_med_l)
max_val=0
for i in range(2,2001):
    D_norm, v1 = stats.ks_2samp(binned_med, stats.t.pdf(center_med, i))
    if v1>max_val:
        max_val = v1
        n=i
ks_arr[3,3] = max_val
print(n)


plt.show()

####################################################################################################################






#######KS tests for different scale factors for norm, cauchy and laplace, first column = scale facotr next 3 columns for wm+(norm, cauchy, laplace order), next 3 for wm-, next 3 for median


#data_arr = df_data.as_matrix()

ks_arr_scale= np.zeros((2500,10))

for i, s in enumerate(np.arange(0.001, 2.501, 0.001)):
    
    ks_arr_scale[i,0] = s
    
    ###p-values for different scale factors for all distributions(weighted mean+)
    D_norm, ks_arr_scale[i,1] = stats.ks_2samp(binned_wmp, stats.norm.pdf(x = center_wmp, scale =s))
    D_norm, ks_arr_scale[i,2] = stats.ks_2samp(binned_wmp, stats.cauchy.pdf(x = center_wmp, scale =s))
    D_norm, ks_arr_scale[i,3] = stats.ks_2samp(binned_wmp, stats.laplace.pdf(x = center_wmp, scale =s))
    
    
    ###p-values for different scale factors for all distributions(weighted mean-)
    D_norm, ks_arr_scale[i,4] = stats.ks_2samp(binned_wmn, stats.norm.pdf(x = center_wmn, scale =s))
    D_norm, ks_arr_scale[i,5] = stats.ks_2samp(binned_wmn, stats.cauchy.pdf(x = center_wmn, scale =s))
    D_norm, ks_arr_scale[i,6] = stats.ks_2samp(binned_wmn, stats.laplace.pdf(x = center_wmn, scale =s))
   
    
#    ###p-values for different scale factors for all distributions(arithmetic mean)
#    D_norm, ks_arr_scale[i,7] = stats.ks_2samp(data_arr[:,2], 'norm', args =(0,s))
#    D_norm, ks_arr_scale[i,8] = stats.ks_2samp(data_arr[:,2], 'cauchy', args =(0,s))
#    D_norm, ks_arr_scale[i,9] = stats.ks_2samp(data_arr[:,2], 'laplace', args =(0,s))

    
    ###p-values for different scale factors for all distributions(median)
    D_norm, ks_arr_scale[i,7] = stats.ks_2samp(binned_med, stats.norm.pdf(x = center_med, scale =s))
    D_norm, ks_arr_scale[i,8] = stats.ks_2samp(binned_med, stats.cauchy.pdf(x = center_med, scale =s))
    D_norm, ks_arr_scale[i,9] = stats.ks_2samp(binned_med, stats.laplace.pdf(x = center_med, scale =s))
    

indices = np.argmax(ks_arr_scale, axis=0)





max_arr = np.zeros((9,2))

for i in range(1,10):
    max_arr[i-1,0]= ks_arr_scale[indices[i],0]
    max_arr[i-1,1]= ks_arr_scale[indices[i],i]
    



plt.plot(center_wmp, z_wmp_g, label = "g")
plt.plot(center_wmp, z_wmp_c, label = "c")
plt.plot(center_wmp, z_wmp_l, label ="l")
#plt.plot(center_wmp, stats.norm.pdf(x = center_wmp, scale =1), label = "scale factor = 1")
plt.plot(center_wmp, z_wmp_t, label = "t")
plt.legend()
plt.savefig("plot4")

plt.show() 
    
plt.plot(center_wmp, binned_wmp)
plt.plot(center_wmp, stats.norm.pdf(center_wmp))
plt.savefig("plot1")

plt.show()

plt.plot(center_wmp, binned_wmp, label = "actual values at bin center")
plt.plot(center_wmp, stats.norm.pdf(center_wmp), label = "scale factor =1")
plt.plot(center_wmp, stats.norm.pdf(x = center_wmp, scale =1.174), label ="scale factor= 1.174(max p value)")
#plt.plot(center_wmp, stats.norm.pdf(x = center_wmp, scale =1), label = "scale factor = 1")
plt.plot(center_wmp, stats.norm.pdf(x = center_wmp, scale =2), label = "scale factor 2")
plt.legend()
plt.savefig("plot2")

plt.show()

plt.plot(ks_arr_scale[:, 0], ks_arr_scale[:,1])
plt.xlabel("scale factors")
plt.ylabel("p-values for gaussian for wm+")
plt.savefig("plot3")
plt.show()


n=2000


######This part finds 3 arrays for t distribution, each for wm+wm- and median, rows = scale factors, columns = dof (n) starting from n=2, ks_arr_t1 for wm+, ks_arr_t2 for wm-, ks_arr_t3 for median

######This makes the compilation really slow.

ks_arr_t1= np.zeros((2500,n+1))
ks_arr_t2= np.zeros((2500,n+1))
ks_arr_t3= np.zeros((2500,n+1))

for i, s in enumerate(np.arange(0.001, 2.501, 0.001)):
    for n in range(2,2001):
        #ks_arr_t[i,0] = s
        D_norm, ks_arr_t1[i,n] = stats.ks_2samp(binned_wmp, stats.t.pdf(x = center_wmp,df = n, scale =s))
        D_norm, ks_arr_t2[i,n] = stats.ks_2samp(binned_wmn, stats.t.pdf(x = center_wmn,df = n, scale =s))
        D_norm, ks_arr_t3[i,n] = stats.ks_2samp(binned_med, stats.t.pdf(x = center_med,df = n, scale =s))
############x1, x2, x3 are the indices of the max values in ks_arr_t1, ks_arr_t2, ks_arr_t3. This way you can also find the scale factor for which max p-value occurs by using 0.001 + (x1.row_number +1)0.01
x1 = np.unravel_index(ks_arr_t1.argmax(), ks_arr_t1.shape)

x2 = np.unravel_index(ks_arr_t2.argmax(), ks_arr_t2.shape)

x3 = np.unravel_index(ks_arr_t3.argmax(), ks_arr_t2.shape)






















'''
###p-values for scal factor =1 for all distributions(weighted mean+)
D_norm, ks_arr[0,0] = stats.kstest(data_arr[:,0], 'norm')
D_norm, ks_arr[0,1] = stats.kstest(data_arr[:,0], 'cauchy')
D_norm, ks_arr[0,2] = stats.kstest(data_arr[:,0], 'laplace')

max_val=0
for i in range(1,2001):
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
for i in range(1,2001):
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
for i in range(1,2001):
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
for i in range(1,2001):
    D_norm, v1 = stats.kstest(data_arr[:,3], 't', args=(i,0,1))
    if v1>max_val:
        max_val = v1
        n=i
ks_arr[3,3] = max_val
print(n)
#######KS tests for different scale factors for norm, cauchy and laplace, first column = scale facotr next 3 columns for wm+(norm, cauchy, laplace order), next 3 for wm-, next 3 for median


data_arr = df_data.as_matrix()

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


for i, s in enumerate(np.arange(0.001, 2.501, 0.01)):
    for n in range(2,2001):
        #ks_arr_t[i,0] = s
        D_norm, ks_arr_t1[i,n] = stats.kstest(data_arr[:,0], 't', args =(n, 0,s))

############x1, x2, x3 are the indices of the max values in ks_arr_t1, ks_arr_t2, ks_arr_t3. This way you can also find the scale factor for which max p-value occurs by using 0.001 + (x1.row_number +1)0.01
x1 = np.unravel_index(ks_arr_t1.argmax(), ks_arr_t1.shape)


ks_arr_t2= np.zeros((250,n+1))

for i, s in enumerate(np.arange(0.001, 2.501, 0.01)):
    for n in range(2,2001):
        #ks_arr_t[i,0] = s
        D_norm, ks_arr_t2[i,n] = stats.kstest(data_arr[:,1], 't', args =(n, 0,s))

x2 = np.unravel_index(ks_arr_t2.argmax(), ks_arr_t2.shape)

ks_arr_t3= np.zeros((250,n+1))

for i, s in enumerate(np.arange(0.001, 2.501, 0.01)):
    for n in range(2,2001):
        #ks_arr_t[i,0] = s
        D_norm, ks_arr_t3[i,n] = stats.kstest(data_arr[:,3], 't', args =(n, 0,s))

x3 = np.unravel_index(ks_arr_t3.argmax(), ks_arr_t2.shape)
'''