import numpy as np
from matplotlib import pyplot as plt

#### Plot results for sparse missing masks ####

# read in data from csv file
data = np.genfromtxt('tests_results_DSM2_SAITS_sparse_1723660984.csv', delimiter=',')

# plot data
plt.figure()
plt.plot(data[:,1], data[:,2], 'o-')
plt.plot(data[:,1], data[:,3], 'o-')
plt.plot(data[:,1], data[:,4], 'o-')
plt.grid(True)
plt.ylim(0,1)
plt.legend(['MAE','RMSE','MRE'])
plt.xlabel('Percent Missing')
plt.savefig('plots/saits_sparse_results.png')

#### Plot results for block missing masks ####

# read in data from csv file
data = np.genfromtxt('tests_results_DSM2_SAITS_block_maskfeat1_1723664882.csv', delimiter=',')

# plot data
plt.figure()
plt.plot(data[:,1], data[:,2], 'o-')
plt.plot(data[:,1], data[:,3], 'o-')
plt.plot(data[:,1], data[:,4], 'o-')
plt.grid(True)
plt.ylim(0,1)
plt.legend(['MAE','RMSE','MRE'])
plt.xlabel('Missing Block Length')
plt.savefig('plots/satits_block_maskfeat1_results.png')

# read in data from csv file
data = np.genfromtxt('tests_results_DSM2_SAITS_block_maskfeat2_1723665153.csv', delimiter=',')

# plot data
plt.figure()
plt.plot(data[:,1], data[:,2], 'o-')
plt.plot(data[:,1], data[:,3], 'o-')
plt.plot(data[:,1], data[:,4], 'o-')
plt.grid(True)
plt.ylim(0,1)
plt.legend(['MAE','RMSE','MRE'])
plt.xlabel('Missing Block Length')
plt.savefig('plots/satits_block_maskfeat2_results.png')

# read in data from csv file
data = np.genfromtxt('tests_results_DSM2_SAITS_block_maskfeat3_1723665387.csv', delimiter=',')

# plot data
plt.figure()
plt.plot(data[:,1], data[:,2], 'o-')
plt.plot(data[:,1], data[:,3], 'o-')
plt.plot(data[:,1], data[:,4], 'o-')
plt.grid(True)
plt.ylim(0,1)
plt.legend(['MAE','RMSE','MRE'])
plt.xlabel('Missing Block Length')
plt.savefig('plots/satits_block_maskfeat3_results.png')

# read in data from csv file
data = np.genfromtxt('tests_results_DSM2_SAITS_block_maskfeat4_1723665630.csv', delimiter=',')

# plot data
plt.figure()
plt.plot(data[:,1], data[:,2], 'o-')
plt.plot(data[:,1], data[:,3], 'o-')
plt.plot(data[:,1], data[:,4], 'o-')
plt.grid(True)
plt.ylim(0,1)
plt.legend(['MAE','RMSE','MRE'])
plt.xlabel('Missing Block Length')
plt.savefig('plots/satits_block_maskfeat4_results.png')
