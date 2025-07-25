"""
Quantum Embedding Methods: LPFET vs DET for Hubbard Model
==========================================================

This script implements and compares two quantum embedding methods:
1. LPFET (Local Potential Functional Embedding Theory)
2. DET (Density Embedding Theory)

For a 6-site Hubbard ring with external potential and varying correlation strength U.


"""
#%%
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
N_mo = 6                    # Number of molecular orbitals (sites)
N_el = 6                    # Number of electrons
N_occ = N_el // 2          # Number of occupied orbitals

# Cluster parameters
N_mo_cl = 2                # Number of cluster orbitals
N_el_cl = 2                # Number of cluster electrons

# Environment parameters
N_el_env = N_el - N_el_cl
N_occ_env = N_el_env // 2

# Physical parameters
t = [1, 1, 1, 1, 1, 1]     # Hopping parameters
v_ext = 1                   # External potential strength
v_ext_array = np.array([-v_ext, 2*v_ext, -2*v_ext, 3*v_ext, -3*v_ext, v_ext])

# Correlation strength values to study
# U_list = np.linspace(0, 10, 10)  # Reduced for clarity
U_list=[1,4,7,10]
# Build quantum many-body basis sets
basis = tools.build_nbody_basis(N_mo, N_el)
a_dag_a = tools.build_operator_a_dagger_a(basis)
basis_cl = tools.build_nbody_basis(N_mo_cl, N_el_cl)
a_dag_a_cl = tools.build_operator_a_dagger_a(basis_cl)


#%%
# ============================================================================
#                           LPFET IMPLEMENTATION
# ============================================================================

def norm_density_LPFET(params):
    """
    LPFET cost function: minimize |n_cluster - n_KS|
    
    Parameters:
    -----------
    params : array
        Local potential parameters
        
    Returns:
    --------
    float : Density difference norm
    """
    global occ_cluster, occ_KS, intermediate_electron_data, U_current
    
    v_hxc = params.copy()
    v_KS = v_hxc + v_ext_array
    
    h_KS = lpfet.h_matrix(N_mo, N_el, t, v_KS, configuration="ring")
    h =  lpfet.h_matrix(N_mo, N_el, t, v_ext_array, configuration="ring")
    
    for impurity_index in range(N_mo):
        # Solve KS equations with permuted Hamiltonian
        h_KS_permuted =  lpfet.switch_sites_matrix(h_KS, impurity_index)
        h_permuted =  lpfet.switch_sites_matrix(h, impurity_index)
        
        epsil, C = scipy.linalg.eigh(h_KS_permuted)
        RDM_KS = C[:, :N_occ] @ C[:, :N_occ].T
        
        # Householder transformation
        P, v = tools.householder_transformation(RDM_KS)
        RDM_KS_HH = P @ RDM_KS @ P
        
        # Diagonalize environment
        n_core, Q = eigh(RDM_KS_HH[N_mo_cl:, N_mo_cl:].copy())
        Q =  lpfet.direct_sum(np.eye(N_mo_cl), Q)
        P_mod = P @ Q
        
        # Build cluster Hamiltonian
        U_HH =  lpfet.u_matrix(N_mo, U_current, delocalized_rep=True, orb_coeffs=P_mod)
        # U_cl = U_HH[:N_mo_cl, :N_mo_cl, :N_mo_cl, :N_mo_cl].copy()
        
        core_energy, h_cl_core, U_cl_core = tools.fh_get_active_space_integrals(
            (P_mod.T) @ h_permuted @ P_mod, U_HH, 
            frozen_indices=[4, 5], active_indices=[0, 1]
        )
        
        # Apply local potential correction
        v_hxc_permuted =  lpfet.switch_sites_vector(v_hxc, impurity_index)
        mu = sum((P_mod[:, 1]**2) * v_hxc_permuted)
        h_cl_core = h_cl_core - np.diag([mu, 0])
        
        # Solve cluster problem
        H_cl = tools.build_hamiltonian_quantum_chemistry(
            h_cl_core, U_cl_core, basis_cl, a_dag_a_cl
        )
        E_cl, Psi_cl = scipy.linalg.eigh(H_cl.toarray())
        
        # Extract densities
        RDM1_cl = tools.build_1rdm_alpha(Psi_cl[:, 0], a_dag_a_cl)
        occ_cluster[impurity_index] = RDM1_cl[0, 0]
        occ_KS[impurity_index] = RDM_KS_HH[0, 0]
    
    # Store convergence data
    N_electron_total = sum(occ_cluster)
    intermediate_electron_data[U_current].append(N_electron_total)
    
    # Return density difference norm
    dens_diff = occ_cluster - occ_KS
    return np.linalg.norm(dens_diff)

#%%
# ============================================================================
#                           DET IMPLEMENTATION
# ============================================================================

def norm_density_DET(params):
    """
    DET cost function: minimize |n_cluster - n_KS| with  globale chemical potential
    
    Parameters:
    -----------
    params : array
        Hxc potential parameters + chemical potential (last element)
        
    Returns:
    --------
    float : Density difference norm
    """
    global occ_cluster_DET, occ_KS_DET, intermediate_electron_data_DET, U_current
    
    v_Hxc = params[:-1].copy()
    mu_DET = params[-1]
    
    h_KS = h_DET + np.diag(v_Hxc)
    
    for impurity_index in range(N_mo):
        # Solve KS equations
        h_KS_permuted =  lpfet.switch_sites_matrix(h_KS, impurity_index)
        epsil, C = scipy.linalg.eigh(h_KS_permuted)
        RDM_KS = C[:, :N_occ] @ C[:, :N_occ].T
        
        # Get embedding orbitals
        C_ht =  lpfet.householder_orbitals(RDM_KS, N_mo_cl)
        
        # Transform integrals
        h_permuted =  lpfet.switch_sites_matrix(h_DET, impurity_index)
        g_permuted =  lpfet.switch_sites_tensor4(g_DET, impurity_index)
        h_Ht, g_Ht = tools.transform_1_2_body_tensors_in_new_basis(
            h_permuted, g_permuted, C_ht
        )
        
        # Active space
        cluster_indices = list(range(N_mo_cl))
        env_occ_indices = [N_mo_cl + i for i in range(N_occ_env)]
        core_energy, h_cl_core, g_cl_core = tools.qc_get_active_space_integrals(
            h_Ht, g_Ht, env_occ_indices, cluster_indices
        )
        
        # Build cluster Hamiltonian with chemical potential
        H_cl = tools.build_hamiltonian_quantum_chemistry(
            h_cl_core - np.diag([mu_DET, 0]), g_cl_core, basis_cl, a_dag_a_cl
        )
        E_cl, Psi_cl = scipy.linalg.eigh(H_cl.toarray())
        
        # Extract densities
        RDM1_cl = tools.build_1rdm_alpha(Psi_cl[:, 0], a_dag_a_cl)
        occ_cluster_DET[impurity_index] = RDM1_cl[0, 0]
        occ_KS_DET[impurity_index] = RDM_KS[0, 0]
    
    # Store convergence data
    N_electron_total = sum(occ_cluster_DET)
    intermediate_electron_data_DET[U_current].append(N_electron_total)
    
    # Return density difference norm
    dens_diff = occ_cluster_DET - occ_KS_DET
    return np.linalg.norm(dens_diff)

#%%
# ============================================================================
#                           ENERGY CALCULATION
# ============================================================================

def calculate_LPFET_energy(v_hxc, U):
    """Calculate total energy using LPFET method."""
    v_KS = v_hxc + v_ext_array
    h_KS =  lpfet.h_matrix(N_mo, N_el, t, v_KS, configuration="ring")
    h =  lpfet.h_matrix(N_mo, N_el, t, v_ext_array, configuration="ring")
    
    sum_site_energy = 0
    for impurity_index in range(N_mo):
        h_KS_permuted =  lpfet.switch_sites_matrix(h_KS, impurity_index)
        h_permuted =  lpfet.switch_sites_matrix(h, impurity_index)
        
        epsil, C = eigh(h_KS_permuted)
        RDM_KS = C[:, :N_occ] @ C[:, :N_occ].T
        
        # Householder transformation
        P, v = tools.householder_transformation(RDM_KS)
        RDM_KS_HH = P @ RDM_KS @ P
        n_core, Q = eigh(RDM_KS_HH[N_mo_cl:, N_mo_cl:].copy())
        Q =  lpfet.direct_sum(np.eye(N_mo_cl), Q)
        P_mod = P @ Q
        
        # Build cluster Hamiltonian for energy
        U_HH =  lpfet.u_matrix(N_mo, U, delocalized_rep=False, orb_coeffs=P_mod)
        U_cl = U_HH[:N_mo_cl, :N_mo_cl, :N_mo_cl, :N_mo_cl].copy()
        
        core_energy, h_cl_core, U_cl_core = tools.fh_get_active_space_integrals(
            (P_mod.T) @ h_KS_permuted @ P_mod, U_HH,
            frozen_indices=[4, 5], active_indices=[0, 1]
        )
        
        h_cl_core = h_cl_core - np.diag([v_hxc[impurity_index], 0])
        H_cl = tools.build_hamiltonian_fermi_hubbard(h_cl_core, U_cl, basis_cl, a_dag_a_cl)
        
        E_cl, Psi_cl = eigh(H_cl.toarray())
        RDM1_cl_free, RDM2_cl_free = tools.build_1rdm_and_2rdm_spin_free(
            Psi_cl[:, 0], a_dag_a_cl
        )
        
        # Energy according to Wouters2016 Eq. 28
        E_fragment = (np.einsum('q,q', h_cl_core[0, :], RDM1_cl_free[0, :]) +
                     0.5 * np.einsum('qrs,qrs', U_cl_core[0, :, :, :], RDM2_cl_free[0, :, :, :]))
        sum_site_energy += E_fragment
    
    return sum_site_energy

def calculate_DET_energy(v_Hxc, mu_DET, U):
    """Calculate total energy using DET method."""
    h_KS = h_DET + np.diag(v_Hxc)
    
    sum_site_energy = 0
    for impurity_index in range(N_mo):
        h_KS_permuted =  lpfet.switch_sites_matrix(h_KS, impurity_index)
        epsil, C = scipy.linalg.eigh(h_KS_permuted)
        RDM_KS = C[:, :N_occ] @ C[:, :N_occ].T
        
        C_ht =  lpfet.householder_orbitals(RDM_KS, N_mo_cl)
        h_permuted =  lpfet.switch_sites_matrix(h_DET, impurity_index)
        g_permuted =  lpfet.switch_sites_tensor4(g_DET, impurity_index)
        h_Ht, g_Ht = tools.transform_1_2_body_tensors_in_new_basis(
            h_permuted, g_permuted, C_ht
        )
        
        cluster_indices = list(range(N_mo_cl))
        env_occ_indices = [N_mo_cl + i for i in range(N_occ_env)]
        core_energy, h_cl_core, g_cl_core = tools.qc_get_active_space_integrals(
            h_Ht, g_Ht, env_occ_indices, cluster_indices
        )
        
        H_cl = tools.build_hamiltonian_quantum_chemistry(
            h_cl_core - np.diag([mu_DET, 0]), g_cl_core, basis_cl, a_dag_a_cl
        )
        E_cl, Psi_cl = scipy.linalg.eigh(H_cl.toarray())
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
    
    return sum_site_energy

#%%
# ============================================================================
#                           MAIN CALCULATION
# ============================================================================

def run_embedding_calculations():
    """
    Main function to run LPFET and DET calculations for all U values
    """
    global occ_cluster, occ_KS, occ_cluster_DET, occ_KS_DET
    global intermediate_electron_data, intermediate_electron_data_DET
    global h_DET, g_DET, U_current
    
    # Initialize storage
    results = {
        'FCI_energies': [],
        'FCI_densities': [],
        'LPFET_energies': [],
        'LPFET_densities': [],
        'DET_energies': [],
        'DET_densities': [],
        'LPFET_potentials': [],
        'DET_potentials': []
    }
    
    # Initialize data storage
    intermediate_electron_data = {U: [] for U in U_list}
    intermediate_electron_data_DET = {U: [] for U in U_list}
    
    # DET-specific setup
    h_DET =  lpfet.h_matrix(N_mo, N_el, t, v_ext_array, configuration="ring")
    g_DET = np.zeros((N_mo, N_mo, N_mo, N_mo))
    
    # Optimization settings
    options_optimizer = {"maxiter": 2000, "ftol": 1e-7}
    initial_guess_LPFET = -v_ext_array
    initial_guess_DET = np.zeros(N_mo + 1)
    
    print("Starting embedding calculations...")
    print("=" * 50)
    
    for U in U_list:
        U_current = U
        print(f"\nProcessing U = {U}")
        print("-" * 30)
        
        # Set up interaction for current U
        for i in range(N_mo):
            g_DET[i, i, i, i] = U
        
        # FCI reference calculation
        print("Computing FCI reference...")
        h =  lpfet.h_matrix(N_mo, N_el, t, v_ext_array, configuration="ring")
        U_operator =  lpfet.u_matrix(N_mo, U)
        H_ref = tools.build_hamiltonian_fermi_hubbard(h, U_operator, basis, a_dag_a)
        E_FCI, Psi_FCI = scipy.sparse.linalg.eigsh(H_ref, which="SA", k=6)
        RDM_FCI = tools.build_1rdm_alpha(Psi_FCI[:, 0], a_dag_a)
        FCI_density = [RDM_FCI[i, i] for i in range(N_mo)]
        
        results['FCI_energies'].append(E_FCI[0])
        results['FCI_densities'].append(FCI_density)
        
        # LPFET calculation
        print("Running LPFET optimization...")
        occ_cluster = np.zeros(N_mo)
        occ_KS = np.zeros(N_mo)
        
        result_LPFET = scipy.optimize.minimize(
            norm_density_LPFET, x0=initial_guess_LPFET, method='L-BFGS-B',
            options=options_optimizer
        )
        
        v_hxc = result_LPFET.x.copy()
        E_LPFET = calculate_LPFET_energy(v_hxc, U)
        
        results['LPFET_energies'].append(E_LPFET)
        results['LPFET_densities'].append(occ_cluster.copy())
        results['LPFET_potentials'].append(v_hxc.copy())
        
        # DET calculation
        print("Running DET optimization...")
        occ_cluster_DET = np.zeros(N_mo)
        occ_KS_DET = np.zeros(N_mo)
        
        result_DET = scipy.optimize.minimize(
            norm_density_DET, x0=initial_guess_DET, method='L-BFGS-B',
            options=options_optimizer
        )
        
        v_Hxc_DET = result_DET.x[:-1]
        mu_DET = result_DET.x[-1]
        E_DET = calculate_DET_energy(v_Hxc_DET, mu_DET, U)
        
        results['DET_energies'].append(E_DET)
        results['DET_densities'].append(occ_cluster_DET.copy())
        results['DET_potentials'].append(v_Hxc_DET.copy())
        
        print(f"Results for U = {U}:")
        print(f"  FCI Energy:   {E_FCI[0]:.6f}")
        print(f"  LPFET Energy: {E_LPFET:.6f}")
        print(f"  DET Energy:   {E_DET:.6f}")
    
    return results

#%%
# ============================================================================
#                           PLOTTING FUNCTIONS
# ============================================================================

def plot_density_comparison(results):
    """Create density comparison plots for all U values."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    axs = axs.flatten()
    
    for i, U in enumerate(U_list):
        if i >= len(axs):
            break
            
        axs[i].plot(range(N_mo), results['FCI_densities'][i], 
                   ls="--", label="FCI", color='black', marker='o', 
                   linewidth=2, markersize=6)
        axs[i].plot(range(N_mo), results['LPFET_densities'][i], 
                   ls="-.", label="LPFET", color='dodgerblue', marker='^', 
                   linewidth=2, markersize=6)
        axs[i].plot(range(N_mo), results['DET_densities'][i], 
                   ls="-.", label="DET", color='red', marker='x', 
                   linewidth=2, markersize=6)
        
        axs[i].set_title(f"U = {U}", fontsize=25)
        axs[i].grid(True)
        if i in [2, 3]:
            axs[i].set_xlabel("Site", fontsize=24)
        if i in [0, 2]:
            axs[i].set_ylabel("Density Per Spin", fontsize=24)
        if i == 3:
            axs[i].legend(loc='upper left', fontsize=20)
        
        axs[i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    
    plt.tight_layout()
    plt.savefig("density_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def plot_energy_comparison(results):
    """Create energy comparison plot."""
    plt.figure(figsize=(10, 8))
    
    plt.plot(U_list, results['FCI_energies'], 
             label="FCI", color='black', linestyle='-', marker='o', 
             linewidth=2, markersize=8)
    plt.plot(U_list, results['LPFET_energies'], 
             label="LPFET", color='dodgerblue', linestyle='--', marker='^', 
             linewidth=2, markersize=8)
    plt.plot(U_list, results['DET_energies'], 
             label="DET", color='red', linestyle='--', marker='s', 
             linewidth=2, markersize=8)
    
    plt.xlabel('U/t', fontsize=25)
    plt.ylabel('Energy', fontsize=25)
    plt.grid(True)
    plt.legend(loc='best', fontsize=20)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    
    plt.tight_layout()
    plt.savefig("energy_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.show()

# def plot_convergence_analysis(results):
#     """Plot convergence of electron count during optimization."""
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
#     max_iterations = 50
    
#     # Plot LPFET convergence
#     for U in U_list:
#         data = intermediate_electron_data[U][:max_iterations]
#         ax1.plot(range(len(data)), data, label=f'U = {U}', 
#                 marker='o', markersize=4, linewidth=2)
    
#     ax1.set_ylabel('Number of Electrons', fontsize=14)
#     ax1.set_title('LPFET Electron Count Convergence', fontsize=16)
#     ax1.legend(fontsize=12)
#     ax1.grid(True, alpha=0.3)
    
#     # Plot DET convergence
#     for U in U_list:
#         data = intermediate_electron_data_DET[U][:max_iterations]
#         ax2.plot(range(len(data)), data, label=f'U = {U}', 
#                 linestyle='--', marker='x', markersize=4, linewidth=2)
    
#     ax2.set_xlabel('Iteration Step', fontsize=14)
#     ax2.set_ylabel('Number of Electrons', fontsize=14)
#     ax2.set_title('DET Electron Count Convergence', fontsize=16)
#     ax2.legend(fontsize=12)
#     ax2.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig("convergence_analysis.pdf", dpi=300, bbox_inches='tight')
#     plt.show()

#%%
# ============================================================================
#                           RUN CALCULATIONS
# ============================================================================

if __name__ == "__main__":
    # Run the calculations
    print("Quantum Embedding Methods: LPFET vs DET")
    print("=======================================")
    
    results = run_embedding_calculations()
    
    # Create plots
    print("\nCreating plots...")
    plot_density_comparison(results)
    plot_energy_comparison(results)
    # plot_convergence_analysis(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CALCULATION SUMMARY")
    print("=" * 60)
    print(f"{'U':<8} {'FCI':<12} {'LPFET':<12} {'DET':<12}")
    print("-" * 60)
    
    for i, U in enumerate(U_list):
        print(f"{U:<8.2f} {results['FCI_energies'][i]:<12.6f} "
              f"{results['LPFET_energies'][i]:<12.6f} "
              f"{results['DET_energies'][i]:<12.6f}")
    
    print("\nCalculations completed successfully!")
    print("Plots saved as PDF files in current directory.")
# %%
