import numpy as np

def direct_sum(A,B):
    zero_matrix_A=np.zeros((A.shape[0],B.shape[1]))
    zero_matrix_B=np.zeros((B.shape[0],A.shape[1]))
    dir_sum=np.block([[A,zero_matrix_A],[zero_matrix_B,B]])
    return(dir_sum)

# Defining the switch_sites_matrix function
def switch_sites_matrix(M, new_impurity):
    M_permuted = M.copy()
    M_permuted[:, [0, new_impurity]] = M_permuted[:, [new_impurity, 0]]
    M_permuted[[0, new_impurity], :] = M_permuted[[new_impurity, 0], :]
    return M_permuted

def switch_sites_tensor4(M, new_impurity):
    M_permuted = M.copy()
    M_permuted[[0, new_impurity],:,:,:] = M_permuted[[new_impurity, 0],:,:,:]
    M_permuted[:,[0, new_impurity],:,:] = M_permuted[:,[new_impurity, 0],:,:]
    M_permuted[:,:,[0, new_impurity],:] = M_permuted[:,:,[new_impurity, 0],:]
    M_permuted[:,:,:,[0, new_impurity]] = M_permuted[:,:,:,[new_impurity, 0]]
    return M_permuted
