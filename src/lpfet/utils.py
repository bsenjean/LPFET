import numpy as np
import sys
import scipy
import quantnbody as qnb

def direct_sum(A,B):
    zero_matrix_A=np.zeros((A.shape[0],B.shape[1]))
    zero_matrix_B=np.zeros((B.shape[0],A.shape[1]))
    dir_sum=np.block([[A,zero_matrix_A],[zero_matrix_B,B]])
    return(dir_sum)

# Defining the switch_site functions
def switch_sites_vector(M, new_impurity):
    M_permuted = M.copy()
    M_permuted[0], M_permuted[new_impurity] = M_permuted[new_impurity], M_permuted[0]
    return M_permuted

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

def householder_orbitals(RDM, N_mo_cl):
    """
    Generate Householder-transformed orbitals for embedding.
    
    Parameters:
    -----------
    RDM : array
        Reduced density matrix
    N_mo_cl : int
        Number of cluster orbitals
        
    Returns:
    --------
    array : Householder orbitals
    """
    P, v = qnb.fermionic.tools.householder_transformation(RDM)
    RDM_ht = P @ RDM @ P
    RDM_ht_env = RDM_ht[N_mo_cl:, N_mo_cl:]
    
    # Separate occupied from virtual orbitals
    occ_env, C_ht_env = scipy.linalg.eigh(RDM_ht_env)
    C_ht = direct_sum(np.eye(N_mo_cl), np.fliplr(C_ht_env))
    
    # Transform back to original basis
    return P @ C_ht


def h_matrix(n_mo, n_elec, t, v, length, width, periodic=False):
    """
    Build the one-body Hamiltonian matrix for the Hubbard model.
    
    Parameters:
    -----------
    n_mo : int
        Number of molecular orbitals
    n_elec : int
        Number of electrons
    t : hopping parameter
    v : array
        On-site potentials.
    length: size of the chain / ladder
    width: width of the ladder
    periodic: periodic or not
        
    Returns:
    --------
    array : One-body Hamiltonian matrix
    """
    tM = np.zeros((n_mo, n_mo))

    # Add on-site potentials
    tM += np.diag(v)
    
    # Build a ladder for the hopping parameter:
    #      LENGTH = 4
    # 0 --- 1 --- 2 --- 3  W
    # |     |     |     |  I
    # 4 --- 5 --- 6 --- 7  D = 3
    # |     |     |     |  T
    # 8 --- 9 --- 10 -- 11 H   

    # length-hopping:
    for w in range(width): # width = 1 --> w = 0
      for l in range(length - 1):
        tM[w*length + l, w*length + l + 1] = tM[w*length + l + 1, w*length + l] = -t
      if periodic: 
        tM[w*length, (w+1)*length - 1] = tM[(w+1)*length - 1, w*length] = -t
    # width-hopping:
    if width > 1:
      for l in range(length):
        for w in range(width - 1): # width == 1 --> range(0) = nothing happens.
          tM[w*length + l, (w+1)*length + l] = tM[(w+1)*length + l, w*length + l] = -t
        if periodic:
          tM[l, (width-1)*length + l] = tM[(width-1)*length + l, l] = -t
    
    # Antiperiodic boundary conditions for a ring
    if periodic:
        if n_elec % 4 == 0 and width==1:
            tM[0, n_mo - 1] = tM[n_mo - 1, 0] = t
    
    return tM

def u_matrix(n_mo, U, delocalized_rep=False, orb_coeffs=None):
    """
    Build the two-body interaction tensor.
    
    Parameters:
    -----------
    n_mo : int
        Number of molecular orbitals
    U : float
        Coulomb repulsion strength
    delocalized_rep : bool
        Whether to use delocalized representation
    orb_coeffs : array
        Orbital coefficients for transformation
        
    Returns:
    --------
    array : Two-body interaction tensor
    """
    UM = np.zeros((n_mo, n_mo, n_mo, n_mo))
    
    if delocalized_rep and orb_coeffs is not None:
        for p in range(n_mo):
            for q in range(n_mo):
                for r in range(n_mo):
                    for s in range(n_mo):
                        UM[p, q, r, s] = U * np.sum(
                            orb_coeffs[:, p] * orb_coeffs[:, q] * 
                            orb_coeffs[:, r] * orb_coeffs[:, s])
    else:
        for i in range(n_mo):
            UM[i, i, i, i] = U
    
    return UM
