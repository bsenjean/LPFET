
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
    P, v = tools.householder_transformation(RDM)
    RDM_ht = P @ RDM @ P
    RDM_ht_env = RDM_ht[N_mo_cl:, N_mo_cl:]
    
    # Separate occupied from virtual orbitals
    occ_env, C_ht_env = scipy.linalg.eigh(RDM_ht_env)
    C_ht = direct_sum(np.eye(N_mo_cl), np.fliplr(C_ht_env))
    
    # Transform back to original basis
    return P @ C_ht


def h_matrix(n_mo, n_elec, t, v, configuration="ring"):
    """
    Build the one-body Hamiltonian matrix for the Hubbard model.
    
    Parameters:
    -----------
    n_mo : int
        Number of molecular orbitals
    n_elec : int
        Number of electrons
    t : array
        Hopping parameters
    v : array
        On-site potentials
    configuration : str
        "ring" or "line" geometry
        
    Returns:
    --------
    array : One-body Hamiltonian matrix
    """
    tM = np.zeros((n_mo, n_mo))
    
    # Nearest neighbor hopping
    for i in range(n_mo - 1):
        tM[i, i + 1] = tM[i + 1, i] = -t[i]
    
    # Periodic boundary conditions for ring
    if configuration == "ring":
        if n_elec % 4 == 2:
            tM[0, n_mo - 1] = tM[n_mo - 1, 0] = -t[-1]
        elif n_elec % 4 == 0:
            tM[0, n_mo - 1] = tM[n_mo - 1, 0] = t[-1]
    
    # Add on-site potentials
    tM += np.diag(v)
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
