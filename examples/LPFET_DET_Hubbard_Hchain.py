"""
Quantum Embedding Methods: LPFET vs DET for Hubbard and Hydrogen Chains
========================================================================

This script implements and compares two quantum embedding methods:
1. LPFET (Local Potential Functional Embedding Theory)
2. DET (Density Embedding Theory)

For a 6-site Hubbard ring with external potential and varying correlation strength U.
For linear hydrogen chains at different interatomic variables using quantum chemistry.


"""

import numpy as np
import os, sys
import scipy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import quantnbody as qnb
import quantnbody.fermionic.tools as tools
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
import lpfet 
import psi4

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
N_mo = 6                    # Number of molecular orbitals or sites
N_el = 6                    # Number of electrons
N_occ = N_el // 2          # Number of occupied orbitals

# Cluster parameters
N_mo_cl = 2                # Number of cluster orbitals
N_el_cl = 2                # Number of cluster electrons

# Environment parameters
N_el_env = N_el - N_el_cl
N_occ_env = N_el_env // 2

system_list = ['Hubbard','Hchain']

# Build quantum many-body basis sets
basis = tools.build_nbody_basis(N_mo, N_el)
a_dag_a = tools.build_operator_a_dagger_a(basis)
basis_cl = tools.build_nbody_basis(N_mo_cl, N_el_cl)
a_dag_a_cl = tools.build_operator_a_dagger_a(basis_cl)

# Classical optimizer criteria:
basinhopping = False
niter_hopping = 10  # parameter used for basinhopping
opt_method = ['L-BFGS-B', 'SLSQP'][0]  # classical optimizer
opt_maxiter = 5000  # number of iterations of the classical optimizer
ftol = 1e-9
gtol = 1e-6

#%%
# ============================================================================
#                           LPFET and DET IMPLEMENTATION
# ============================================================================

def norm_density(params):
    """
    LPFET or DET cost function: minimize |n_cluster - n_KS|
    
    Parameters:
    -----------
    params : array
        Local potential parameters
      if DET: additional global chemical potential
        
    Returns:
    --------
    float : Density difference norm
    """
    global occ_cluster, occ_KS, sum_site_energy
    
    if len(params)==N_mo:
    # LPFET
       v_Hxc = params.copy()
    elif len(params)==N_mo+1:
    # DET
       v_Hxc = params.copy()[:-1]

    h_KS = h + np.diag(v_Hxc)
    epsil, C = scipy.linalg.eigh(h_KS)
    RDM = C[:, :N_occ] @ C[:, :N_occ].T
    sum_site_energy = 0
    for impurity_index in range(N_mo):
        # Permute Hamiltonian to set current site as impurity
        h_KS_permuted = lpfet.switch_sites_matrix(h_KS, impurity_index)
        
        # Solve KS equations
        epsil, C = scipy.linalg.eigh(h_KS_permuted)
        RDM = C[:, :N_occ] @ C[:, :N_occ].T
        
        # Get Householder orbitals for embedding
        C_Ht = lpfet.householder_orbitals(RDM, N_mo_cl)
        
        # Permute integrals for current impurity
        h_permuted = lpfet.switch_sites_matrix(h, impurity_index)
        g_permuted = lpfet.switch_sites_tensor4(g, impurity_index)
        
        # Transform to embedding basis
        h_Ht, g_Ht = tools.transform_1_2_body_tensors_in_new_basis(
            h_permuted, g_permuted, C_Ht
        )
        
        # Define active space
        cluster_indices = list(range(N_mo_cl))
        env_occ_indices = [N_mo_cl + i for i in range(N_occ_env)]
        
        # Get active space integrals
        core_energy, h_cl_core, g_cl_core = tools.qc_get_active_space_integrals(
            h_Ht, g_Ht, env_occ_indices, cluster_indices
        )

        # Apply local potential correction
        v_Hxc_permuted = lpfet.switch_sites_vector(v_Hxc, impurity_index)
        if len(params)==N_mo: mu = sum((C_Ht[:, 1]**2) * v_Hxc_permuted)
        elif len(params)==N_mo+1: mu = params[-1]

        # Build and solve cluster Hamiltonian
        H_cl = tools.build_hamiltonian_quantum_chemistry(
            h_cl_core - np.diag([mu, 0]), g_cl_core, basis_cl, a_dag_a_cl
        )
        E_cl, Psi_cl = scipy.linalg.eigh(H_cl.toarray())
        
        # Extract densities
        RDM1_cl = tools.build_1rdm_alpha(Psi_cl[:, 0], a_dag_a_cl)
        occ_cluster[impurity_index] = RDM1_cl[0, 0]
        occ_KS[impurity_index] = RDM[0, 0]

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
    
    # Return density difference norm
    dens_diff = occ_cluster - occ_KS
    return np.linalg.norm(dens_diff)

#%%
# ============================================================================
#                           MAIN CALCULATION
# ============================================================================

def run_embedding_calculations():
    """
    Main function to run LPFET and DET calculations for all variables
    """
    global occ_cluster, occ_KS, sum_site_energy
    global h, g
    
    # Storage for results
    results = {
        'FCI_energies': [],
        'FCI_densities': [],
        'LPFET_energies': [],
        'LPFET_densities': [],
        'LPFET_potentials': [],
        'LPFET_conv': [],
        'DET_energies': [],
        'DET_densities': [],
        'DET_potentials': [],
        'DET_conv': [],
        'variables': variables
    }
    
    print("Starting embedding calculations...")
    print("=" * 60)
    
    for var in variables:
        print(f"\nProcessing {system} with variable = {var}")
        print("-" * 30)
        
        # Generate geometry and get integrals

        if system == 'Hubbard':
          t = [1, 1, 1, 1, 1, 1]     # Hopping parameters
          v_ext = 1                   # External potential strength
          v_ext_array = np.array([-v_ext, 2*v_ext, -2*v_ext, 3*v_ext, -3*v_ext, v_ext])
          h = lpfet.h_matrix_2D(N_mo, N_el, t, v_ext_array,link_params=[1,3], configuration="ring")
          g = np.zeros((N_mo, N_mo, N_mo, N_mo))
          for i in range(N_mo):
            g[i, i, i, i] = var
        elif system == 'Hchain':
          geometry = tools.generate_h_chain_geometry(N_mo, var)
          psi4.core.clean()
          psi4.core.clean_variables()
          psi4.core.clean_options()
          psi4.core.set_output_file("output_Psi4.txt", False)
          geometry += '\n' + 'symmetry c1'
          molecule = psi4.geometry(geometry)
          molecule.set_molecular_charge(0)
          psi4.set_options({'basis' : 'sto-3g','fail_on_maxiter':False, 'maxiter':0})
          # Realizing a generic HF calculation ======
          E_HF, scf_wfn = psi4.energy('scf', molecule=molecule, return_wfn=True )
          E_nuc = molecule.nuclear_repulsion_energy()
          mints = psi4.core.MintsHelper(scf_wfn.basisset())
          S = np.asarray(mints.ao_overlap())
          h_AO = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
          g_AO = np.asarray(mints.ao_eri()).reshape(( np.shape( h_AO )[0],
                                                       np.shape( h_AO )[0],
                                                       np.shape( h_AO )[0],
                                                       np.shape( h_AO )[0] ))
          # Orthogonalize to OAO basis (Löwdin symmetric orthogonalization)
          eigenvalues, eigenvectors = np.linalg.eigh(S)
          S_half = eigenvectors @ np.diag(eigenvalues**(-0.5)) @ eigenvectors.T
          h, g = tools.transform_1_2_body_tensors_in_new_basis(
              h_AO, g_AO, S_half
          )
        
        # FCI reference calculation
        print("Computing FCI reference...")
        H_ref = tools.build_hamiltonian_quantum_chemistry(h, g, basis, a_dag_a)
        E_FCI, Psi_FCI = scipy.sparse.linalg.eigsh(H_ref, which="SA", k=6)
        RDM_FCI = tools.build_1rdm_alpha(Psi_FCI[:, 0], a_dag_a)
        FCI_density = [RDM_FCI[i, i] for i in range(N_mo)]
        
        if system == 'Hchain': E_FCI[0] += E_nuc
        results['FCI_energies'].append(E_FCI[0])
        results['FCI_densities'].append(FCI_density)
        
        # LPFET calculation
        print("Running LPFET optimization...")
        occ_cluster = np.zeros(N_mo)
        occ_KS = np.zeros(N_mo)
        
        #initial_guess_LPFET = np.zeros(N_mo)
        if system == 'Hubbard': 
           initial_guess_LPFET = - np.diag(h)
           initial_guess_DET = np.zeros(N_mo + 1)  # +1 for chemical potential
        if system == 'Hchain':
           initial_guess_LPFET = np.zeros(N_mo)
           initial_guess_DET = np.zeros(N_mo + 1)  # +1 for chemical potential

        if opt_method == "L-BFGS-B": opt_options = {'maxiter': opt_maxiter, 'ftol': ftol, 'gtol': gtol}
        if opt_method == "SLSQP": opt_options = {'maxiter': opt_maxiter, 'ftol': ftol}
        if not basinhopping:                
           result_LPFET = scipy.optimize.minimize(
               norm_density, x0=initial_guess_LPFET, method=opt_method,
               options=opt_options)
        else:
           result_LPFET = scipy.optimize.basinhopping(
               norm_density, x0=initial_guess_LPFET,niter=niter_hopping,
                                                minimizer_kwargs={'method': opt_method})

        print(result_LPFET)
        print("cluster:",occ_cluster)
        print("KS:",occ_KS)
        v_Hxc = result_LPFET['x']
        E_LPFET = sum_site_energy
        if system == 'Hchain': E_LPFET += E_nuc
        results['LPFET_energies'].append(E_LPFET)
        results['LPFET_densities'].append(occ_cluster.copy())
        results['LPFET_potentials'].append(v_Hxc.copy())
        results['LPFET_conv'].append(result_LPFET.fun)
        
        # DET calculation
        print("Running DET optimization...")
        
        occ_cluster = np.zeros(N_mo)
        occ_KS = np.zeros(N_mo)
        if not basinhopping:
           result_DET = scipy.optimize.minimize(
               norm_density, x0=initial_guess_DET, method=opt_method,
               options=opt_options)
        else:
           result_DET = scipy.optimize.basinhopping(
               norm_density, x0=initial_guess_DET,niter=niter_hopping,
                                                minimizer_kwargs={'method': opt_method})

        print(result_DET)
        print("cluster:",occ_cluster)
        print("KS:",occ_KS)
        v_Hxc_DET = result_DET.x[:-1]
        mu_DET = result_DET.x[-1]
        E_DET = sum_site_energy
        if system == 'Hchain': E_DET += E_nuc
        results['DET_energies'].append(E_DET)
        results['DET_densities'].append(occ_cluster.copy())
        results['DET_potentials'].append(v_Hxc_DET.copy())
        results['DET_conv'].append(result_DET.fun)
        
        print(f"Results for {system}, variable = {var}:" )
        print(f"  FCI Energy:   {E_FCI[0]:.6f} hartree")
        print(f"  LPFET Energy: {E_LPFET:.6f} hartree")
        print(f"  DET Energy:   {E_DET:.6f} hartree")
    
    return results

#%%
# ============================================================================
#                           PLOTTING FUNCTIONS
# ============================================================================

def plot_energy_comparison(results):
    """Create energy comparison plot."""
    plt.figure(figsize=(10, 8))
    
    plt.plot(results['variables'], results['FCI_energies'], 
             label="FCI", color='black', linestyle='-', marker='o', 
             linewidth=2, markersize=6)
    plt.plot(results['variables'], results['LPFET_energies'], 
             label="LPFET", color='dodgerblue', linestyle='--', marker='s', 
             linewidth=2, markersize=6)
    plt.plot(results['variables'], results['DET_energies'], 
             label="DET", color='red', linestyle='--', marker='^', 
             linewidth=2, markersize=6)
    
    if system=='Hubbard':
      x = '$U/t$'
      y = 'Energy/$t$'
    elif system=='Hchain':
      x = 'Bond distance [\\AA]'
      y = 'Energy [hartree]'
    
    plt.xlabel(x, fontsize=27)
    plt.ylabel(y, fontsize=27)
    plt.grid(True)
    plt.legend(loc='best', fontsize=28)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    
    plt.tight_layout()
    plt.savefig("energy_comparison_"+system+".pdf", dpi=300, bbox_inches='tight')
    plt.show()

def plot_individual_densities(results):
    """Create individual density plots for each distance."""
    for i, var in enumerate(results['variables']):
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
        
        if system=='Hubbard':
          x = 'Site'
          y = 'Density per spin'
          title = '$U/t$ = {}'.format(var)
        elif system=='Hchain':
          x = 'OAO Index'
          y = 'Density per spin'
          title = '$R$ = {} \\AA'.format(var)

        plt.xlabel(x, fontsize=17)
        plt.ylabel(y, fontsize=17)
        plt.title(title, fontsize=17)
        plt.grid(True)
        plt.legend()
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        #plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        
        plt.tight_layout()
        plt.savefig(f"density_"+system+f"{var:.2f}.pdf", dpi=300, bbox_inches='tight')
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

    for system in system_list:
      if system == 'Hubbard':
        #variables = np.linspace(0,40,20)
        variables = [2.0,3.0,4.0,5.0,6.0,7.0,8.0]
      elif system == 'Hchain':
        variables = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.5,2.0,2.5,3.0,3.1,3.2,3.3,3.4]
        #variables = [0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]


      print("Quantum Embedding Methods: LPFET vs DET for {}".format(system))
      print("============================================================")
      
      # Run the calculations
      results = run_embedding_calculations()
      
      # Create plots
      plot_results(results)
      
      # Print summary
      print("\n" + "=" * 60)
      print("CALCULATION SUMMARY")
      print("=" * 60)
      print("-" * 60)
      
      f = open(system+'.dat', 'a')
      print(f"{'Variable':<12} {'FCI':<12} {'LPFET':<12} {'DET':<12} {'LPFET conv':<16} {'DET conv':<16}")
      print(f"{'Variable':<12} {'FCI':<12} {'LPFET':<12} {'DET':<12} {'LPFET conv':<16} {'DET conv':<16}", file = f)
      for i, var in enumerate(results['variables']):
          print(f"{var:<12.1f} {results['FCI_energies'][i]:<12.6f} "
                f"{results['LPFET_energies'][i]:<12.6f} "
                f"{results['DET_energies'][i]:<12.6f}"
                f"{results['LPFET_conv'][i]:<16.12f}"
                f"{results['DET_conv'][i]:<16.12f}")
          print(f"{var:<12.1f} {results['FCI_energies'][i]:<12.6f} "
                f"{results['LPFET_energies'][i]:<12.6f} "
                f"{results['DET_energies'][i]:<12.6f}"
                f"{results['LPFET_conv'][i]:<16.12f}"
                f"{results['DET_conv'][i]:<16.12f}",file=f)

      f.close()
      
      print("\nCalculations completed successfully!")
      print("Plots saved as PDF files in current directory.")
# %%
