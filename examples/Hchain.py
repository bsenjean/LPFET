import numpy as np 
import os
import sys 
import psi4
import scipy 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import quantnbody as qnb
import quantnbody.fermionic.tools as tools
import lpfet

def norm_density(params):
    """
    Cost function used in LPFET (cluster occupation = KS occupation).
    Adapted to the symmetry of a linear Hydrogen chain, i.e. left part = right part of the molecule.
    """

    global occ_cluster
    global occ_KS

    # Use symmetry for the KS potential (depends on the system!)
    v_KS = np.zeros(2*(len(params)))
    for i in range(len(params)):
      v_KS[i] = params[i]
      v_KS[-(i+1)] = params[i]

    h_OAO_vKS = h_OAO + np.diag(v_KS)
    sum_site_energy = 0
    for impurity_index in range(N_mo):
        # permutation is done on the Hamiltonian, that's all
        h_permuted = lpfet.switch_sites_matrix(h_OAO_vKS, impurity_index)
        epsil, C = scipy.linalg.eigh(h_permuted)
        RDM_OAO = C[:, :N_occ] @ C[:, :N_occ].T 
        # Get the householder orbitals:
        # The orbitals are sorted as 1) cluster 2) occupied environment 3) virtual environment
        C_ht = lpfet.Householder_orbitals(RDM_OAO,N_mo_cl)
        # Don't forget to permute the sites in the 2-body integrals!
        h_OAO_permuted = lpfet.switch_sites_matrix(h_OAO,impurity_index)
        g_OAO_permuted = lpfet.switch_sites_tensor4(g_OAO,impurity_index)
        # Compute the 1- and 2-body integrals
        h_Ht, g_Ht = tools.transform_1_2_body_tensors_in_new_basis( h_OAO_permuted, g_OAO_permuted, C_ht )
        # Use the Frozen-core (active space) approximation.
        cluster_indices = [ i for i in range(N_mo_cl) ]
        env_occ_indices = [ N_mo_cl + i for i in range(N_occ_env) ]
        core_energy, h_cl_core, g_cl_core = tools.qc_get_active_space_integrals(h_Ht, g_Ht, env_occ_indices, cluster_indices)
        # Build the Hamiltonian of the cluster using the active space and frozen-core orbitals
        H_cl = tools.build_hamiltonian_quantum_chemistry( h_cl_core, g_cl_core, basis_cl, a_dag_a_cl )
        # Solve the Hamiltonian
        E_cl, Psi_cl = scipy.linalg.eigh(H_cl.A)
        # Extract the 1RDM of the ground state and the occupation of the impurity site
        RDM1_cl = tools.build_1rdm_alpha(Psi_cl[:,0], a_dag_a_cl)
        occ_cluster[impurity_index] = RDM1_cl[0,0]
        occ_KS[impurity_index] = RDM_OAO[0,0]

    # impose the last site occupation to match the number of electrons
    penalty = np.abs(occ_cluster[N_mo-1] - (N_el//2 - sum(occ_cluster[:N_mo-1])))
    occ_cluster[N_mo-1] = N_el//2 - sum(occ_cluster[:N_mo-1])
    dens_diff_list = occ_cluster - occ_KS
    Dens_diff = np.linalg.norm(dens_diff_list) + penalty

    return Dens_diff 

# Operators for the full system
N_mo = 6
N_el = 6
N_occ = N_el//2 
basis = tools.build_nbody_basis(N_mo,N_el)
a_dag_a = tools.build_operator_a_dagger_a(basis)

# Operators for the embedding cluster
N_mo_cl = 2
N_el_cl = 2
basis_cl = tools.build_nbody_basis(N_mo_cl,N_el_cl)
a_dag_a_cl= tools.build_operator_a_dagger_a(basis_cl)

# environment:
N_el_env = N_el-N_el_cl
N_occ_env = N_el_env//2

# setting initial parameters 
options_optimizer = {"maxiter": 2000, "ftol": 1e-6} 
initial_guess = np.zeros(N_mo//2)

# Lists to store results
converged_densities = [] 
converged_densities_KS = [] 
FCI_densities = []
FCI_energies = []
E_HF_list = []
E_tot = []

Distance=[0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.5,2.0,2.5,3.0]
for R in Distance:
    geometry = tools.generate_h_chain_geometry(N_mo, R)
    overlap_AO, h_AO, g_AO, C_RHF, E_HF, E_nuc = tools.get_info_from_psi4( geometry,basisset="sto-3g")
    E_HF_list.append(E_HF)

    # Construct the OAO basis by Lowdin Symmetric Orthogonalization:
    S = overlap_AO
    # Step 1: Diagonalize the overlap matrix S
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    # Step 2: Construct the S^(1/2) matrix
    S_half = eigenvectors @ np.diag(eigenvalues**(-1/2)) @ eigenvectors.T

    # Transform one-body and two-body integrals to the OAO basis
    h_OAO, g_OAO = tools.transform_1_2_body_tensors_in_new_basis(h_AO, g_AO, S_half)

    # Build the FCI Hamiltonian
    H_ref = tools.build_hamiltonian_quantum_chemistry( h_OAO, g_OAO , basis, a_dag_a)
    # Solve the FCI problem using eigsh (sparse eigenvalue solver)
    E_FCI, Psi_FCI = scipy.sparse.linalg.eigsh(H_ref, which="SA", k=6)
    # Calculate the 1-RDM for the alpha spin
    RDM_FCI = tools.build_1rdm_alpha(Psi_FCI[:,0], a_dag_a)
    # Collect the diagonal elements of the 1-RDM (FCI density)
    FCI_density = []
    for i in range(N_mo):
        FCI_density.append(RDM_FCI[i, i])
    FCI_densities.append(FCI_density)
    FCI_energies.append(E_FCI[0] + E_nuc)

    # Initialization
    occ_cluster = np.zeros(N_mo)
    occ_KS = np.zeros(N_mo)

    print(f"\nStarting optimization for R = {R}") 
    Optimized = scipy.optimize.minimize(norm_density, x0=initial_guess, method='L-BFGS-B', options=options_optimizer) 
    print(Optimized)
    v_KS = np.zeros(2*(len(Optimized.x)))
    for i in range(len(Optimized.x)):
      v_KS[i] = Optimized.x[i]
      v_KS[-(i+1)] = Optimized.x[i]
    # Store results after optimization 
    converged_densities.append(occ_cluster) 
    converged_densities_KS.append(occ_KS) 
    print("\nOptimization completed.")
    print("Final converged density:", occ_cluster)
    print("Final converged KS density:", occ_KS)
    print("FCI density:",FCI_density)
    print("Final KS potentials:", v_KS)
 
    h_OAO_vKS = h_OAO + np.diag(v_KS)

    # Now compute the energy:
    sum_site_energy = 0
    for impurity_index in range(N_mo):
        # permutation is done on the Hamiltonian, that's all
        h_permuted = lpfet.switch_sites_matrix(h_OAO_vKS, impurity_index)
        epsil, C = scipy.linalg.eigh(h_permuted)
        RDM_OAO = C[:, :N_occ] @ C[:, :N_occ].T
        # Get the householder orbitals:
        # The orbitals are sorted as 1) cluster 2) occupied environment 3) virtual environment
        C_ht = lpfet.Householder_orbitals(RDM_OAO,N_mo_cl)
        # Don't forget to permute the sites in the 2-body integrals!
        h_OAO_permuted = lpfet.switch_sites_matrix(h_OAO,impurity_index)
        g_OAO_permuted = lpfet.switch_sites_tensor4(g_OAO,impurity_index)
        # Compute the 1- and 2-body integrals
        h_Ht, g_Ht = tools.transform_1_2_body_tensors_in_new_basis( h_OAO_permuted, g_OAO_permuted, C_ht )
        # Use the Frozen-core (active space) approximation.
        cluster_indices = [ i for i in range(N_mo_cl) ]
        env_occ_indices = [ N_mo_cl + i for i in range(N_occ_env) ]
        core_energy, h_cl_core, g_cl_core = tools.qc_get_active_space_integrals(h_Ht, g_Ht, env_occ_indices, cluster_indices)
        # Build the Hamiltonian of the cluster using the active space and frozen-core orbitals
        H_cl = tools.build_hamiltonian_quantum_chemistry( h_cl_core, g_cl_core, basis_cl, a_dag_a_cl )
        E_cl, Psi_cl = scipy.linalg.eigh(H_cl.A)
        RDM1_cl_free, RDM2_cl_free = tools.build_1rdm_and_2rdm_spin_free( Psi_cl[:,0], a_dag_a_cl )
        E_fragment = np.einsum('q,q', (h_cl_core[0,:]), RDM1_cl_free[0,:])+(1./2)*np.einsum('qrs,qrs', g_cl_core[0,:,:,:], RDM2_cl_free[0,:,:,:])
        sum_site_energy += E_fragment
    E_tot.append(sum_site_energy + E_nuc)

#####################################################
#                    PLOTS                          #
#####################################################
for i in range(len(Distance)):
    plt.figure(figsize=(8, 6))

    plt.plot(range(N_mo), FCI_densities[i], ls="--", label="FCI", color='black', marker='o')
    plt.plot(range(N_mo), converged_densities[i], ls="-.", label="Cluster", color='r', marker='x')
    plt.plot(range(N_mo), converged_densities_KS[i], ls=":", label="KS", color='b', marker='+')

    plt.title("Interatomic distance of R = {}".format(Distance[i]))
    plt.xlabel("Hydrogen", fontsize=17)
    plt.ylabel("Orbital occupation", fontsize=17)
    plt.grid(True)

    plt.legend()
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))
    plt.show()

plt.plot(Distance, FCI_energies, label="FCI",color='black', linestyle='-', marker='o')
plt.plot(Distance, E_tot, label="Embedding Energy",color='dodgerblue', linestyle='--', marker='s')
plt.xlabel('Distance [angstrom]', fontsize=17)
plt.ylabel('Energy [hartree]', fontsize=17)
plt.grid(True)
plt.legend()
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()

