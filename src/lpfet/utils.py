
def direct_sum(A,B):
    zero_matrix_A=np.zeros((A.shape[0],B.shape[1]))
    zero_matrix_B=np.zeros((B.shape[0],A.shape[1]))
    dir_sum=np.block([[A,zero_matrix_A],[zero_matrix_B,B]])
    return(dir_sum)




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



def u_matrix(n_mo, U, nearest_neighbor_interactions=False, alpha=0, delocalized_rep=False, orb_coeffs=None):
    UM = np.zeros((n_mo, n_mo, n_mo, n_mo))

    if nearest_neighbor_interactions:
        for i in range(n_mo):
            UM[i, i, i, i] = U
            for j in range(i + 1, n_mo):
                UM[i, i, j, j] = UM[j, j, i, i] = alpha * U

    else:
        if delocalized_rep:
            for p in range(n_mo):
                for q in range(n_mo):
                    for r in range(n_mo):
                        for s in range(n_mo):
                            UM[p, q, r, s] = U * np.sum(
                                orb_coeffs[:, p] * orb_coeffs[:, q] * orb_coeffs[:, r] * orb_coeffs[:, s])
        else:
            for i in range(n_mo):
                UM[i, i, i, i] = U

    return UM




def u_matrix_non_interact_bath(n_mo,U):
    UM = np.zeros((n_mo, n_mo, n_mo, n_mo))
    UM[0,0,0,0] = U
    return UM




def h_matrix(n_mo, n_elec, t, v, configuration="ring", BLA_mode=False, alpha=0):
    tM = np.zeros((n_mo, n_mo))

    for i in range(n_mo):
        for j in range(i + 1, n_mo):
            if j == i + 1:
                if configuration == "line" and BLA_mode:
                    if j % 2 != 0:
                        tM[i, j] = tM[j, i] = -t[i] * (1 + alpha)
                    else:
                        tM[i, j] = tM[j, i] = -t[i] * (1 - alpha)
                else:
                    tM[i, j] = tM[j, i] = -1* t[i]
            else:
                tM[i, j] = tM[j, i] = 0

    if configuration == "ring":
        if n_elec % 4 == 2:
            # Periodic BC
            for i in range(len(t)):
                tM[0, n_mo - 1] = tM[n_mo - 1, 0] = -1* t[i]
        elif n_elec % 4 == 0:
            # Antiperiodic BC
            tM[0, n_mo - 1] = tM[n_mo - 1, 0] = t[-1]

    elif configuration == "line":
        pass

    tM += np.diag(v)

    return tM





def switch_sites_vector(v, new_impurity):
    """Permute vector to set new_impurity as element 0."""
    v_permuted = v.copy()
    v_permuted[0], v_permuted[new_impurity] = v_permuted[new_impurity], v_permuted[0]
    return v_permuted






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

