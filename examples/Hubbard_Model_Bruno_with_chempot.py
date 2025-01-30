import numpy as np 
import os
import sys 
import scipy 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import quantnbody as qnb
import quantnbody.fermionic.tools as tools
import lpfet

def vHxc_without_chempot_opt(params):
    """
    Cost function used in LPFET (cluster occupation = KS occupation).
    """

    global occ_cluster
    global occ_KS
    global v_Hxc


    v_Hxc[selected_vHxc_sites] = params.copy()
    #v_Hxc = params.copy()
    h_KS = lpfet.h_matrix(N_mo, N_el, t, v_ext_array + v_Hxc, configuration="ring") 
    epsil, C = scipy.linalg.eigh(h_KS)
    RDM_KS_initial = C[:, :N_occ] @ C[:, :N_occ].T 
    occ_KS = np.diag(RDM_KS_initial)
    dens_diff_list = []
    for impurity_index in selected_vHxc_sites:
    #for impurity_index in range(N_mo):
        # permutation is done on the Hamiltonian, that's all
        h_KS_permuted = lpfet.switch_sites_matrix(h_KS, impurity_index)
        epsil, C = scipy.linalg.eigh(h_KS_permuted)
        RDM_KS = C[:, :N_occ] @ C[:, :N_occ].T 
        # Get the householder orbitals:
        # The orbitals are sorted as 1) cluster 2) occupied environment 3) virtual environment
        C_ht = lpfet.Householder_orbitals(RDM_KS,N_mo_cl)
        h_permuted = lpfet.switch_sites_matrix(h,impurity_index)
        # U is the same everywhere, no need to permute... Could be generalized to different U using U_pqrs instead of U.
        # U_permuted = lpfet.switch_sites_tensor4(U_operator,impurity_index)
        # Building the one-body and two-body transformed parts
        h_Ht = C_ht.T@h_permuted@C_ht
        g_Ht = lpfet.u_matrix(N_mo, U, delocalized_rep=True, orb_coeffs=C_ht)
        # Use the Frozen-core (active space) approximation.
        cluster_indices = [ i for i in range(N_mo_cl) ]
        env_occ_indices = [ N_mo_cl + i for i in range(N_occ_env) ]
        # this work *******************
        core_energy, h_cl_core, g_cl_core = tools.qc_get_active_space_integrals(C_ht.T@h_KS_permuted@C_ht, g_Ht, env_occ_indices, cluster_indices)
        h_cl_core = h_cl_core - np.diag([v_Hxc[impurity_index], 0])
        # *****************************
        # this doesn't ****************
        #core_energy, h_cl_core, g_cl_core = tools.qc_get_active_space_integrals(h_Ht, g_Ht, frozen_indices=env_occ_indices, active_indices=cluster_indices)
        # *****************************
        # Build the Hamiltonian of the cluster using the active space and frozen-core orbitals
        H_cl = tools.build_hamiltonian_quantum_chemistry( h_cl_core, g_cl_core, basis_cl, a_dag_a_cl )
        # Solve the Hamiltonian
        E_cl, Psi_cl = scipy.linalg.eigh(H_cl.toarray())
        # Extract the 1RDM of the ground state and the occupation of the impurity site
        RDM1_cl = tools.build_1rdm_alpha(Psi_cl[:,0], a_dag_a_cl)
        occ_cluster[impurity_index] = RDM1_cl[0,0]
        dens_diff_list.append(occ_cluster[impurity_index] - occ_KS[impurity_index])

    Dens_diff = np.linalg.norm(dens_diff_list)

    return Dens_diff 

def chempot_opt(param,h_cl,g_cl):

    global occ_cluster
    global occ_KS

    # Normally it starts with the optimal v_Hxc determined by the other optimized loop.
    # The last site start with v_Hxc[-1] = chempot determined previously in this loop.
    # the h_cl and g_cl are the 1-body and 2-body integrals from the orbitals coming from this initial v_Hxc.
    # note that the permutation has been done already, the impurity site is the first site.

    H_cl = tools.build_hamiltonian_quantum_chemistry( h_cl - np.diag([param[0],0]), g_cl, basis_cl, a_dag_a_cl )
    # Solve the Hamiltonian
    E_cl, Psi_cl = scipy.linalg.eigh(H_cl.toarray())
    # Extract the 1RDM of the ground state and the occupation of the impurity site
    RDM1_cl = tools.build_1rdm_alpha(Psi_cl[:,0], a_dag_a_cl)
    occ_cluster[selected_chempot_site] = RDM1_cl[0,0]

    Dens_diff = np.linalg.norm(N_el//2 - sum(occ_cluster))

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
options_optimizer = {"maxiter": 2000, "ftol": 1e-8} 
initial_guess = np.zeros(N_mo-1)
chempot = 0

t=np.array([1,1,1,1,1,1])
v_ext=1
v_ext_array=np.array([-v_ext,2*v_ext,-2*v_ext,3*v_ext,-3*v_ext,v_ext])
h = lpfet.h_matrix(N_mo, N_el, t, v_ext_array, configuration="ring") 
U_list=np.linspace(0,15,5)
U_list=[4.0]
selected_chempot_site = 5
selected_vHxc_sites = [i for i in range(N_mo)]
selected_vHxc_sites.remove(selected_chempot_site)

# Lists to store results
converged_densities = [] 
converged_densities_KS = [] 
FCI_densities = []
FCI_energies = []
E_HF_list = []
E_tot = []

for it, U in enumerate(U_list):

    U_operator=lpfet.u_matrix(N_mo,U)
    H_ref=tools.build_hamiltonian_fermi_hubbard(h,U_operator,basis,a_dag_a)
    # Solve the FCI problem using eigsh (sparse eigenvalue solver)
    E_FCI, Psi_FCI = scipy.sparse.linalg.eigsh(H_ref, which="SA", k=6)
    # Calculate the 1-RDM for the alpha spin
    RDM_FCI = tools.build_1rdm_alpha(Psi_FCI[:,0], a_dag_a)
    # Collect the diagonal elements of the 1-RDM (FCI density)
    FCI_density = []
    for i in range(N_mo):
        FCI_density.append(RDM_FCI[i, i])
    FCI_densities.append(FCI_density)
    FCI_energies.append(E_FCI[0])

    # Initialization
    occ_cluster = np.zeros(N_mo)
    occ_KS = np.zeros(N_mo)

    print("\n\n"+"#"*50)
    print(f"\nStarting optimization for U = {U}") 
    print("\n"+"#"*50+"\n\n")
    v_Hxc = np.zeros(N_mo)
    SCF_ITER = 0

    # First start with the embedding on all but the selected chempot site, to initialize the occ_cluster values:
    dens_diff = 1
    #dens_diff = vHxc_without_chempot_opt(np.zeros(N_mo-1))
    #dens_diff = vHxc_without_chempot_opt(np.zeros(N_mo))

    while dens_diff > 1e-6 and SCF_ITER < 20:

      print("*"*50)
      print("Opt of the Hxc potential. v_Hxc = ",v_Hxc)
      print("*"*50)
      optimization = scipy.optimize.minimize(vHxc_without_chempot_opt, x0=v_Hxc[selected_vHxc_sites], method='L-BFGS-B', options=options_optimizer)
      v_Hxc[selected_vHxc_sites] = optimization.x
      #optimization = scipy.optimize.minimize(vHxc_without_chempot_opt, x0=v_Hxc, method='L-BFGS-B', options=options_optimizer)
      #v_Hxc = optimization.x
      print(optimization)
      dens_diff = np.linalg.norm(occ_cluster - occ_KS)
      print("occ cluster:",occ_cluster,sum(occ_cluster))
      print("occ KS     :",occ_KS)
      print("dens_diff  :",dens_diff)

      if dens_diff < 1e-6: break

      #######################################################################
      # before optimizing chempot, one needs to construct H_cl for the selected chempot site
      h_KS = lpfet.h_matrix(N_mo, N_el, t, v_ext_array + v_Hxc, configuration="ring")
      epsil, C = scipy.linalg.eigh(h_KS)
      # permutation is done on the Hamiltonian, that's all
      h_KS_permuted = lpfet.switch_sites_matrix(h_KS, selected_chempot_site)
      epsil, C = scipy.linalg.eigh(h_KS_permuted)
      RDM_KS = C[:, :N_occ] @ C[:, :N_occ].T
      # Get the householder orbitals:
      # The orbitals are sorted as 1) cluster 2) occupied environment 3) virtual environment
      C_ht = lpfet.Householder_orbitals(RDM_KS,N_mo_cl)
      h_permuted = lpfet.switch_sites_matrix(h,selected_chempot_site)
      # U is the same everywhere, no need to permute... Could be generalized to different U using U_pqrs instead of U.
      # U_permuted = lpfet.switch_sites_tensor4(U_operator,impurity_index)
      # Building the one-body and two-body transformed parts
      h_Ht = C_ht.T@h_permuted@C_ht
      g_Ht = lpfet.u_matrix(N_mo, U, delocalized_rep=True, orb_coeffs=C_ht)
      # Use the Frozen-core (active space) approximation.
      cluster_indices = [ i for i in range(N_mo_cl) ]
      env_occ_indices = [ N_mo_cl + i for i in range(N_occ_env) ]
      # this work *******************
      core_energy, h_cl_core, g_cl_core = tools.qc_get_active_space_integrals(C_ht.T@h_KS_permuted@C_ht, g_Ht, env_occ_indices, cluster_indices)
      h_cl_core = h_cl_core - np.diag([v_Hxc[selected_chempot_site], 0])
      #######################################################################

      print("*"*50)
      print("Opt of the chemical potential. v_Hxc = ",v_Hxc)
      print("*"*50)
      optimization = scipy.optimize.minimize(chempot_opt, x0=[v_Hxc[selected_chempot_site]], args=(h_cl_core,g_cl_core), method='L-BFGS-B', options=options_optimizer) 
      print(optimization)
      v_Hxc[selected_chempot_site] = optimization.x[0]
      dens_diff = np.linalg.norm(occ_cluster - occ_KS)
      print("occ cluster:",occ_cluster,sum(occ_cluster))
      print("occ KS     :",occ_KS)
      print("dens_diff  :",dens_diff)

      SCF_ITER += 1

    # Store results after optimization 
    converged_densities.append(occ_cluster) 
    converged_densities_KS.append(occ_KS) 
    print("\nOptimization completed.")
    print("Final converged density:", occ_cluster)
    print("Final converged KS density:", occ_KS)
    print("Final Hxc potentials:", v_Hxc)
 
    h_KS = lpfet.h_matrix(N_mo, N_el, t, v_ext_array + v_Hxc, configuration="ring") 

    # Now compute the energy:
    sum_site_energy = 0
    occ_cluster_test = np.zeros(N_mo)
    for impurity_index in range(N_mo):
        # permutation is done on the Hamiltonian, that's all
        h_KS_permuted = lpfet.switch_sites_matrix(h_KS, impurity_index)
        epsil,C = scipy.linalg.eigh(h_KS_permuted)
        RDM_KS = C[:,:N_occ]@C[:,:N_occ].T
        # Get the householder orbitals:
        # The orbitals are sorted as 1) cluster 2) occupied environment 3) virtual environment
        C_ht = lpfet.Householder_orbitals(RDM_KS,N_mo_cl)
        h_permuted = lpfet.switch_sites_matrix(h,impurity_index)
        # U_permuted = lpfet.switch_sites_tensor4(U_operator,impurity_index)
        # Building the one-body and two-body transformed parts
        h_Ht = C_ht.T@h_permuted@C_ht
        g_Ht = lpfet.u_matrix(N_mo, U, delocalized_rep=True, orb_coeffs=C_ht)
        # Use the Frozen-core (active space) approximation.
        cluster_indices = [ i for i in range(N_mo_cl) ]
        env_occ_indices = [ N_mo_cl + i for i in range(N_occ_env) ]
        # this work *******************
        core_energy, h_cl_core, g_cl_core = tools.qc_get_active_space_integrals(C_ht.T@h_KS_permuted@C_ht, g_Ht, env_occ_indices, cluster_indices)
        h_cl_core = h_cl_core - np.diag([v_Hxc[impurity_index], 0])
        # *****************************
        # this doesn't ****************
        #core_energy, h_cl_core, g_cl_core = tools.qc_get_active_space_integrals(h_Ht, g_Ht, frozen_indices=env_occ_indices, active_indices=cluster_indices)
        # *****************************
        #building the cluster many-electron Hamiltonien 
        H_cl = tools.build_hamiltonian_quantum_chemistry(h_cl_core,g_cl_core,basis_cl,a_dag_a_cl)
        #Solve the many-electron Hamiltonien 
        E_cl,Psi_cl= scipy.linalg.eigh(H_cl.toarray())
        RDM1_cl = tools.build_1rdm_alpha(Psi_cl[:,0], a_dag_a_cl)
        occ_cluster_test[impurity_index] = RDM1_cl[0,0]
        RDM1_cl_free, RDM2_cl_free = tools.build_1rdm_and_2rdm_spin_free( Psi_cl[:,0], a_dag_a_cl )
        E_fragment = 0.5*np.einsum('q,q', (h_Ht[0,:N_mo_cl] + h_cl_core[0,:]), RDM1_cl_free[0,:])+(1./2)*np.einsum('qrs,qrs', g_Ht[0,:N_mo_cl,:N_mo_cl,:N_mo_cl], RDM2_cl_free[0,:,:,:])
        sum_site_energy += E_fragment
    print("Final cluster occupation:",occ_cluster_test,sum(occ_cluster_test))
    E_tot.append(sum_site_energy)


#####################################################
#                    PLOTS                          #
#####################################################


for i in range(len(U_list)):
    plt.figure(figsize=(8, 6))

    plt.plot(range(N_mo), FCI_densities[i], ls="--", label="FCI", color='black', marker='o')
    plt.plot(range(N_mo), converged_densities[i], ls="-.", label="Cluster", color='r', marker='x')
    plt.plot(range(N_mo), converged_densities_KS[i], ls=":", label="KS", color='b', marker='+')

    plt.title("Interatomic distance of U = {}".format(U_list[i]))
    plt.xlabel("Site", fontsize=17)
    plt.ylabel("Orbital occupation", fontsize=17)
    plt.grid(True)

    plt.legend()
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))
    plt.show()

#plt.rc('font',  family='serif') 
#plt.rc('font',  size='14') 
#plt.rc('xtick', labelsize='x-large')
#plt.rc('ytick', labelsize='x-large') 
#plt.rc('lines', linewidth='2')
#plt.rcParams.update({ "text.usetex": True})

plt.plot(U_list, FCI_energies, label="FCI",color='black', linestyle='-', marker='o')
plt.plot(U_list, E_tot, label="Embedding Energy",color='dodgerblue', linestyle='--', marker='s')
plt.xlabel('Correlation ', fontsize=17)
plt.ylabel('Energy ', fontsize=17)
plt.grid(True)
plt.legend()
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()
