import quantnbody as qnb
import scipy
import numpy as np
from .utils import direct_sum

def Householder_orbitals(RDM,N_mo_cl):
    """
    Compute the cluster and (occupied/virtuals) environment orbitals
    from the input 1RDM.
    The orbitals are sorted as 
       - 1) cluster 
       - 2) occupied environment 
       - 3) virtual environment
    """

    # Householder transformation:
    P, v = qnb.fermionic.tools.householder_transformation(RDM)
    RDM_ht = P @ RDM @ P
    RDM_ht_env = RDM_ht[ N_mo_cl:, N_mo_cl: ]
    # Separate occupied from virtual orbitals of the environment:
    occ_env, C_ht_env = scipy.linalg.eigh(RDM_ht_env)
    # recombine the orbitals:
    C_ht = direct_sum(np.eye(N_mo_cl), np.fliplr(C_ht_env))
    # Transform back to the original basis:
    C_ht = P@C_ht

    return C_ht
