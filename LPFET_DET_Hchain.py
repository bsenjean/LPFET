"""
Quantum Embedding Methods: LPFET vs DET for Hydrogen Chains
============================================================

This script implements and compares two quantum embedding methods:
1. LPFET (Local Potential Functional Embedding Theory)  
2. DET (Density Embedding Theory)

For linear hydrogen chains at different interatomic distances using quantum chemistry.


"""

import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import quantnbody as qnb
import quantnbody.fermionic.tools as tools
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
import lpfet 





# Set up plotting parameters
plt.rc('font', family='serif', size=14)
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')
plt.rc('lines', linewidth=2)
plt.rcParams.update({"text.usetex": True})

#%%
# ============================================================================
#                           SYSTEM PARAMETERS
# ============================================================================

# System parameters
N_mo = 6                    # Number of molecular orbitals
N_el = 6                    # Number of electrons
N_occ = N_el // 2          # Number of occupied orbitals

# Cluster parameters
N_mo_cl = 2                # Number of cluster orbitals
N_el_cl = 2                # Number of cluster electrons

# Environment parameters
N_el_env = N_el - N_el_cl
N_occ_env = N_el_env // 2

# Distances to study (in Angstroms)
distances = [0.2,0.4,0.6,0.7, 1.0, 1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.5,3]# Representative distances for clarity

# Build quantum many-body basis sets
basis = tools.build_nbody_basis(N_mo, N_el)
a_dag_a = tools.build_operator_a_dagger_a(basis)
basis_cl = tools.build_nbody_basis(N_mo_cl, N_el_cl)
a_dag_a_cl = tools.build_operator_a_dagger_a(basis_cl)

# Optimization parameters
options_optimizer = {"maxiter": 2000, "ftol": 1e-6}

#%%
# ============================================================================
#                           LPFET IMPLEMENTATION
# ============================================================================

def norm_density_LPFET(params):
    """
    LPFET cost function: minimize |n_cluster - n_KS|
    Uses molecular symmetry for linear hydrogen chains.
    
    Parameters:
    -----------
    params : array
        Local potential parameters (half due to symmetry)
        
    Returns:
    --------
    float : Density difference norm
    """
    global occ_cluster, occ_KS, h_OAO, g_OAO
    
    # Use symmetry for the KS potential (linear chain symmetry)
    v_Hxc = np.zeros(2 * len(params))
    for i in range(len(params)):
        v_Hxc[i] = params[i]
        v_Hxc[-(i+1)] = params[i]
    
    h_OAO_vKS = h_OAO + np.diag(v_Hxc)
    
    for impurity_index in range(N_mo):
        # Permute Hamiltonian to set current site as impurity
        h_permuted = lpfet.switch_sites_matrix(h_OAO_vKS, impurity_index)
        
        # Solve KS equations
        epsil, C = scipy.linalg.eigh(h_permuted)
        RDM_OAO = C[:, :N_occ] @ C[:, :N_occ].T
        
        # Get Householder orbitals for embedding
        C_ht = lpfet.householder_orbitals(RDM_OAO, N_mo_cl)
        
        # Permute integrals for current impurity
        h_OAO_permuted = lpfet.switch_sites_matrix(h_OAO, impurity_index)
        g_OAO_permuted = lpfet.switch_sites_tensor4(g_OAO, impurity_index)
        
        # Transform to embedding basis
        h_Ht, g_Ht = tools.transform_1_2_body_tensors_in_new_basis(
            h_OAO_permuted, g_OAO_permuted, C_ht
        )
        
        # Define active space
        cluster_indices = list(range(N_mo_cl))
        env_occ_indices = [N_mo_cl + i for i in range(N_occ_env)]
        
        # Get active space integrals
        core_energy, h_cl_core, g_cl_core = tools.qc_get_active_space_integrals(
            h_Ht, g_Ht, env_occ_indices, cluster_indices
        )
        
        # Build and solve cluster Hamiltonian
        H_cl = tools.build_hamiltonian_quantum_chemistry(
            h_cl_core, g_cl_core, basis_cl, a_dag_a_cl
        )
        E_cl, Psi_cl = scipy.linalg.eigh(H_cl.toarray())
        
        # Extract densities
        RDM1_cl = tools.build_1rdm_alpha(Psi_cl[:, 0], a_dag_a_cl)
        occ_cluster[impurity_index] = RDM1_cl[0, 0]
        occ_KS[impurity_index] = RDM_OAO[0, 0]
    
    # Return density difference norm
    dens_diff = occ_cluster - occ_KS
    return np.linalg.norm(dens_diff)

#%%
# ============================================================================
#                           DET IMPLEMENTATION
# ============================================================================
#%%
# ============================================================================
#                           DET IMPLEMENTATION
# ============================================================================

def norm_density_DET(params_DET):
    """
    DET cost function: minimize |ρ_cluster - ρ_KS| with chemical potential
    
    Parameters:
    -----------
    params_DET : array
        Hxc potential parameters + chemical potential (last element)
        
    Returns:
    --------
    float : Density difference norm
    """
    global occ_cluster_DET, occ_KS_DET, h_DET, g_DET
    
    # Extract potentials and chemical potential
    v_Hxc_DET = np.zeros(2 * (len(params_DET) - 1))
    for i in range(len(params_DET) - 1):
        v_Hxc_DET[i] = params_DET[i]
        v_Hxc_DET[-(i+1)] = params_DET[i]
    mu = params_DET[-1]
    
    h_KS_DET = h_DET + np.diag(v_Hxc_DET)
    
    for impurity_index in range(N_mo):
        # Permute and solve KS equations
        h_KS_permuted_DET = switch_sites_matrix(h_KS_DET, impurity_index)
        epsil, C = scipy.linalg.eigh(h_KS_permuted_DET)
        RDM_KS_DET = C[:, :N_occ] @ C[:, :N_occ].T
        
        # Get embedding orbitals
        C_ht_DET = lpfet.householder_orbitals(RDM_KS_DET, N_mo_cl)
        
        # Permute integrals
        h_permuted_DET =lpfet.switch_sites_matrix(h_DET, impurity_index)
        g_permuted_DET = lpfet.switch_sites_tensor4(g_DET, impurity_index)
        
        # Transform to embedding basis
        h_Ht_DET, g_Ht_DET = tools.transform_1_2_body_tensors_in_new_basis(
            h_permuted_DET, g_permuted_DET, C_ht_DET
        )
        
        # Active space
        cluster_indices = list(range(N_mo_cl))
        env_occ_indices = [N_mo_cl + i for i in range(N_occ_env)]
        
        core_energy, h_cl_core_DET, g_cl_core_DET = tools.qc_get_active_space_integrals(
            h_Ht_DET, g_Ht_DET, env_occ_indices, cluster_indices
        )
        
        # Build cluster Hamiltonian with chemical potential
        H_cl_DET = tools.build_hamiltonian_quantum_chemistry(
            h_cl_core_DET - np.diag([mu, 0]), g_cl_core_DET, basis_cl, a_dag_a_cl
        )
        E_cl, Psi_cl_DET = scipy.linalg.eigh(H_cl_DET.toarray())
        
        # Extract densities
        RDM1_cl_DET = tools.build_1rdm_alpha(Psi_cl_DET[:, 0], a_dag_a_cl)
        occ_cluster_DET[impurity_index] = RDM1_cl_DET[0, 0]
        occ_KS_DET[impurity_index] = RDM_KS_DET[0, 0]
    
    dens_diff_DET = occ_cluster_DET - occ_KS_DET
    return np.linalg.norm(dens_diff_DET)



#%%# ============================================================================
#                           ENERGY CALCULATIONS
# ============================================================================  


def calculate_LPFET_energy(v_Hxc, E_nuc):
    """Calculate total energy using LPFET method."""
    h_OAO_vKS = h_OAO + np.diag(v_Hxc)
    sum_site_energy = 0
    
    for impurity_index in range(N_mo):
        h_permuted = lpfet.switch_sites_matrix(h_OAO_vKS, impurity_index)
        epsil, C = scipy.linalg.eigh(h_permuted)
        RDM_OAO = C[:, :N_occ] @ C[:, :N_occ].T 
        
        # Embedding orbitals
        C_ht = householder_orbitals(RDM_OAO, N_mo_cl)
        
        # Transform integrals
        h_OAO_permuted = lpfet.switch_sites_matrix(h_OAO, impurity_index)
        g_OAO_permuted = lpfet.switch_sites_tensor4(g_OAO, impurity_index)
        h_Ht, g_Ht = tools.transform_1_2_body_tensors_in_new_basis(
            h_OAO_permuted, g_OAO_permuted, C_ht
        )
        
        # Active space
        cluster_indices = list(range(N_mo_cl))
        env_occ_indices = [N_mo_cl + i for i in range(N_occ_env)]
        core_energy, h_cl_core, g_cl_core = tools.qc_get_active_space_integrals(
            h_Ht, g_Ht, env_occ_indices, cluster_indices
        )
        
        # Fix: Create the diagonal matrix properly for the 2x2 cluster
        h_cl_core_corrected = h_cl_core - np.diag([v_Hxc[impurity_index], 0])
        
        # Solve cluster problem
        H_cl = tools.build_hamiltonian_quantum_chemistry(
            h_cl_core_corrected, g_cl_core, basis_cl, a_dag_a_cl
        )
        E_cl, Psi_cl = scipy.linalg.eigh(H_cl.toarray())
        
        # Calculate fragment energy using Wouters2016 formula (Eq. 28)
        RDM1_cl_free, RDM2_cl_free = tools.build_1rdm_and_2rdm_spin_free(
            Psi_cl[:, 0], a_dag_a_cl
        )
        E_fragment = (0.5 * np.einsum('q,q', 
                                     (h_Ht[0, :N_mo_cl] + h_cl_core[0, :]), 
                                     RDM1_cl_free[0, :]) +
                     0.5 * np.einsum('qrs,qrs', 
                                    g_Ht[0, :N_mo_cl, :N_mo_cl, :N_mo_cl], 
                                    RDM2_cl_free[0, :, :, :]))
        sum_site_energy += E_fragment
    
    return sum_site_energy + E_nuc

def calculate_DET_energy(v_Hxc_DET, mu, E_nuc):
    """Calculate total energy using DET method."""
    h_KS_DET = h_DET + np.diag(v_Hxc_DET)
    sum_site_energy_DET = 0
    
    for impurity_index in range(N_mo):
        h_KS_permuted_DET = lpfet.switch_sites_matrix(h_KS_DET, impurity_index)
        epsil, C = scipy.linalg.eigh(h_KS_permuted_DET)
        RDM_KS_DET = C[:, :N_occ] @ C[:, :N_occ].T
        
        C_ht_DET = lpfet.householder_orbitals(RDM_KS_DET, N_mo_cl)
        h_permuted_DET = lpfet.switch_sites_matrix(h_DET, impurity_index)
        g_permuted_DET = lpfet.switch_sites_tensor4(g_DET, impurity_index)
        h_Ht_DET, g_Ht_DET = tools.transform_1_2_body_tensors_in_new_basis(
            h_permuted_DET, g_permuted_DET, C_ht_DET
        )
        
        cluster_indices = list(range(N_mo_cl))
        env_occ_indices = [N_mo_cl + i for i in range(N_occ_env)]
        core_energy_DET, h_cl_core_DET, g_cl_core_DET = tools.qc_get_active_space_integrals(
            h_Ht_DET, g_Ht_DET, env_occ_indices, cluster_indices
        )
        
        H_cl_DET = tools.build_hamiltonian_quantum_chemistry(
            h_cl_core_DET - np.diag([mu, 0]), g_cl_core_DET, basis_cl, a_dag_a_cl
        )
        E_cl_DET, Psi_cl_DET = scipy.linalg.eigh(H_cl_DET.toarray())
        RDM1_cl_free_DET, RDM2_cl_free_DET = tools.build_1rdm_and_2rdm_spin_free(
            Psi_cl_DET[:, 0], a_dag_a_cl
        )
        
        E_fragment_DET = (0.5 * np.einsum('q,q', 
                                         (h_Ht_DET[0, :N_mo_cl] + h_cl_core_DET[0, :]),
                                         RDM1_cl_free_DET[0, :]) +
                         0.5 * np.einsum('qrs,qrs',
                                        g_Ht_DET[0, :N_mo_cl, :N_mo_cl, :N_mo_cl],
                                        RDM2_cl_free_DET[0, :, :, :]))
        sum_site_energy_DET += E_fragment_DET
    
    return sum_site_energy_DET + E_nuc

#%%
# ============================================================================
#                           MAIN CALCULATION
# ============================================================================

def run_embedding_calculations():
    """
    Main function to run LPFET and DET calculations for all distances
    """
    global occ_cluster, occ_KS, occ_cluster_DET, occ_KS_DET
    global h_OAO, g_OAO, h_DET, g_DET
    
    # Storage for results
    results = {
        'FCI_energies': [],
        'FCI_densities': [],
        'LPFET_energies': [],
        'LPFET_densities': [],
        'DET_energies': [],
        'DET_densities': [],
        'distances': distances
    }
    
    print("Starting embedding calculations...")
    print("=" * 50)
    
    for R in distances:
        print(f"\nProcessing distance R = {R} Å")
        print("-" * 30)
        
        # Generate geometry and get integrals
        geometry = tools.generate_h_chain_geometry(N_mo, R)
        overlap_AO, h_AO, g_AO, C_RHF, E_HF, E_nuc = tools.get_info_from_psi4(
            geometry, basisset="sto-3g"
        )
        
        # Orthogonalize to OAO basis (Löwdin symmetric orthogonalization)
        S = overlap_AO
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        S_half = eigenvectors @ np.diag(eigenvalues**(-0.5)) @ eigenvectors.T
        h_OAO, g_OAO = tools.transform_1_2_body_tensors_in_new_basis(
            h_AO, g_AO, S_half
        )
        
        # Set DET integrals (same as h_OAO, g_OAO for this system)
        h_DET, g_DET = h_OAO.copy(), g_OAO.copy()
        
        # FCI reference calculation
        print("Computing FCI reference...")
        H_ref = tools.build_hamiltonian_quantum_chemistry(h_OAO, g_OAO, basis, a_dag_a)
        E_FCI, Psi_FCI = scipy.sparse.linalg.eigsh(H_ref, which="SA", k=6)
        RDM_FCI = tools.build_1rdm_alpha(Psi_FCI[:, 0], a_dag_a)
        FCI_density = [RDM_FCI[i, i] for i in range(N_mo)]
        
        results['FCI_energies'].append(E_FCI[0] + E_nuc)
        results['FCI_densities'].append(FCI_density)
        
        # LPFET calculation
        print("Running LPFET optimization...")
        occ_cluster = np.zeros(N_mo)
        occ_KS = np.zeros(N_mo)
        
        initial_guess_LPFET = np.zeros(N_mo // 2)
        result_LPFET = scipy.optimize.minimize(
            norm_density_LPFET, x0=initial_guess_LPFET, method='L-BFGS-B',
            options=options_optimizer
        )
        
        # Construct symmetric potential
        v_Hxc = np.zeros(2 * len(result_LPFET.x))
        for i in range(len(result_LPFET.x)):
            v_Hxc[i] = result_LPFET.x[i]
            v_Hxc[-(i+1)] = result_LPFET.x[i]
        
        E_LPFET = calculate_LPFET_energy(v_Hxc, E_nuc)
        results['LPFET_energies'].append(E_LPFET)
        results['LPFET_densities'].append(occ_cluster.copy())
        
        # DET calculation
        print("Running DET optimization...")
        occ_cluster_DET = np.zeros(N_mo)
        occ_KS_DET = np.zeros(N_mo)
        
        initial_guess_DET = np.zeros(N_mo // 2 + 1)  # +1 for chemical potential
        result_DET = scipy.optimize.minimize(
            norm_density_DET, x0=initial_guess_DET, method='L-BFGS-B',
            options=options_optimizer
        )
        
        # Construct DET potential
        v_Hxc_DET = np.zeros(2 * (len(result_DET.x) - 1))
        for i in range(len(result_DET.x) - 1):
            v_Hxc_DET[i] = result_DET.x[i]
            v_Hxc_DET[-(i+1)] = result_DET.x[i]
        mu_DET = result_DET.x[-1]
        
        E_DET = calculate_DET_energy(v_Hxc_DET, mu_DET, E_nuc)
        results['DET_energies'].append(E_DET)
        results['DET_densities'].append(occ_cluster_DET.copy())
        
        print(f"Results for R = {R}:")
        print(f"  FCI Energy:   {E_FCI[0] + E_nuc:.6f} hartree")
        print(f"  LPFET Energy: {E_LPFET:.6f} hartree")
        print(f"  DET Energy:   {E_DET:.6f} hartree")
    
    return results

#%%
# ============================================================================
#                           PLOTTING FUNCTIONS
# ============================================================================

# def plot_density_comparison(results):
#     """Create density comparison plots for all distances."""
#     fig, axs = plt.subplots(2, 2, figsize=(14, 12))
#     axs = axs.flatten()
    
#     for i, R in enumerate(results['distances']):
#         axs[i].plot(range(N_mo), results['FCI_densities'][i], 
#                    ls="--", label="FCI", color='black', marker='o', 
#                    linewidth=2, markersize=6)
#         axs[i].plot(range(N_mo), results['LPFET_densities'][i], 
#                    ls="-.", label="LPFET", color='dodgerblue', marker='x', 
#                    linewidth=2, markersize=6)
#         axs[i].plot(range(N_mo), results['DET_densities'][i], 
#                    ls="-.", label="DET", color='red', marker='s', 
#                    linewidth=2, markersize=6)
        
#         axs[i].set_title(f"R = {R} Å", fontsize=25)
#         axs[i].grid(True)
#         if i in [2, 3]:
#             axs[i].set_xlabel("Hydrogen Atom", fontsize=25)
#         if i in [0, 2]:
#             axs[i].set_ylabel("Density per Spin", fontsize=25)
#         if i == 0:
#             axs[i].legend(loc='best', fontsize=24)
        
#         axs[i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    
#     plt.tight_layout()
#     plt.savefig("density_comparison_hchains.pdf", dpi=300, bbox_inches='tight')
#     plt.show()

def plot_energy_comparison(results):
    """Create energy comparison plot."""
    plt.figure(figsize=(10, 8))
    
    plt.plot(results['distances'], results['FCI_energies'], 
             label="FCI", color='black', linestyle='-', marker='o', 
             linewidth=2, markersize=6)
    plt.plot(results['distances'], results['LPFET_energies'], 
             label="LPFET", color='dodgerblue', linestyle='--', marker='s', 
             linewidth=2, markersize=6)
    plt.plot(results['distances'], results['DET_energies'], 
             label="DET", color='red', linestyle='--', marker='^', 
             linewidth=2, markersize=6)
    
    plt.xlabel('Distance [Å]', fontsize=27)
    plt.ylabel('Energy [hartree]', fontsize=27)
    plt.grid(True)
    plt.legend(loc='best', fontsize=28)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    plt.tight_layout()
    plt.savefig("energy_comparison_hchains.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def plot_individual_densities(results):
    """Create individual density plots for each distance."""
    for i, R in enumerate(results['distances']):
        plt.figure(figsize=(8, 6))
        plt.plot(range(N_mo), results['FCI_densities'][i], 
                ls="--", label="FCI", color='black', marker='o', 
                linewidth=2, markersize=6)
        plt.plot(range(N_mo), results['LPFET_densities'][i], 
                ls="-.", label="LPFET", color='dodgerblue', marker='x', 
                linewidth=2, markersize=6)
        plt.plot(range(N_mo), results['DET_densities'][i], 
                ls="-.", label="DET", color='red', marker='s', 
                linewidth=2, markersize=6)
        
        plt.xlabel("Site", fontsize=17)
        plt.ylabel("Density per spin", fontsize=17)
        plt.grid(True)
        plt.title(f"Density Profile at R = {R} Å", fontsize=20)
        plt.legend()
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        
        plt.tight_layout()
        plt.savefig(f"density_R_{R:.1f}A.pdf", dpi=300, bbox_inches='tight')
        plt.show()

def plot_results(results):
    """Create all plots."""
    # print("Creating density comparison plots...")
    # plot_density_comparison(results)
    
    print("Creating energy comparison plot...")
    plot_energy_comparison(results)
    
    print("Creating individual density plots...")
    plot_individual_densities(results)
    
    print("All plots created successfully!")

#%%
# ============================================================================
#                           RUN CALCULATIONS
# ============================================================================

if __name__ == "__main__":
    print("Quantum Embedding Methods: LPFET vs DET for Hydrogen Chains")
    print("============================================================")
    
    # Run the calculations
    results = run_embedding_calculations()
    
    # Create plots
    plot_results(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CALCULATION SUMMARY")
    print("=" * 60)
    print(f"{'Distance (Å)':<12} {'FCI':<12} {'LPFET':<12} {'DET':<12}")
    print("-" * 60)
    
    for i, R in enumerate(results['distances']):
        print(f"{R:<12.1f} {results['FCI_energies'][i]:<12.6f} "
              f"{results['LPFET_energies'][i]:<12.6f} "
              f"{results['DET_energies'][i]:<12.6f}")
    
    print("\nCalculations completed successfully!")
    print("Plots saved as PDF files in current directory.")
# %%
