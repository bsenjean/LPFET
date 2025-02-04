import numpy as np 
import os
import sys 
import scipy 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import quantnbody as qnb
import quantnbody.fermionic.tools as tools
import lpfet

def norm_density(params):
    """
    Cost function used in LPFET (cluster occupation = KS occupation).
    """

    global occ_cluster
    global occ_KS

    v_Hxc = params.copy()

    h_KS = lpfet.h_matrix(N_mo, N_el, t, v_ext_array + v_Hxc, configuration="ring") 
    for impurity_index in range(N_mo):

        h_KS_permuted = lpfet.switch_sites_matrix(h_KS, impurity_index)
        epsil, C = scipy.linalg.eigh(h_KS_permuted)
        RDM_KS = C[:, :N_occ] @ C[:, :N_occ].T 
        # Get the householder orbitals:
        # The orbitals are sorted as 1) cluster 2) occupied environment 3) virtual environment
        C_ht = lpfet.Householder_orbitals(RDM_KS,N_mo_cl)

        # Compute the 1- and 2-body integrals
        h_Ht = C_ht.T@h_KS_permuted@C_ht
        g_Ht = lpfet.u_matrix(N_mo, U, delocalized_rep=True, orb_coeffs=C_ht)
        # Use the Frozen-core (active space) approximation.
        cluster_indices = [ i for i in range(N_mo_cl) ]
        env_occ_indices = [ N_mo_cl + i for i in range(N_occ_env) ]
        g_Ht_NIB = np.zeros(g_Ht.shape)
        g_Ht_NIB[0,0,0,0] = g_Ht[0,0,0,0]
        core_energy, h_cl_core, g_cl_core = tools.qc_get_active_space_integrals(h_Ht, g_Ht_NIB, env_occ_indices, cluster_indices)
        h_cl_core = h_cl_core - np.diag([v_Hxc[impurity_index], 0])

        # Build the Hamiltonian of the cluster using the active space and frozen-core orbitals
        H_cl = tools.build_hamiltonian_quantum_chemistry( h_cl_core, g_cl_core, basis_cl, a_dag_a_cl )
        # Solve the Hamiltonian
        E_cl, Psi_cl = scipy.linalg.eigh(H_cl.toarray())
        # Extract the 1RDM of the ground state and the occupation of the impurity site
        RDM1_cl = tools.build_1rdm_alpha(Psi_cl[:,0], a_dag_a_cl)
        occ_cluster[impurity_index] = RDM1_cl[0,0]
        occ_KS[impurity_index] = RDM_KS[0,0]

    dens_diff_list = occ_cluster - occ_KS
    Dens_diff = np.linalg.norm(dens_diff_list)

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
t=np.array([1,1,1,1,1,1])
v_ext=1
v_ext_array=np.array([-v_ext,2*v_ext,-2*v_ext,3*v_ext,-3*v_ext,v_ext])
U_list=np.linspace(0,15,16)
basin_hopping = False
opt_method = ["L-BFGS-B","SLSQP"][1]
init = ["zero","vext"][0]
if init == "zero":
  initial_guess = np.zeros(N_mo)
elif init == "vext":
  initial_guess = -v_ext_array
#initial_guess = np.random.rand(N_mo)

# Lists to store results
converged_densities = [] 
converged_densities_KS = [] 
FCI_densities = []
FCI_energies = []
E_HF_list = []
E_tot = []

h =lpfet.h_matrix(N_mo, N_el, t, v_ext_array, configuration="ring")

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

    print(f"\nStarting optimization for U = {U}") 
    if not basin_hopping:
      Optimized = scipy.optimize.minimize(norm_density, x0=initial_guess, method=opt_method) 
    else:
      Optimized = scipy.optimize.basinhopping(norm_density, x0=initial_guess, minimizer_kwargs={"method":opt_method},niter=100) 
    print(Optimized)
    v_Hxc = Optimized.x
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

        epsil,C=scipy.linalg.eigh(h_KS_permuted)
        RDM_KS= (C[:,:N_occ])@(C[:,:N_occ].T)
        # Get the householder orbitals:
        # The orbitals are sorted as 1) cluster 2) occupied environment 3) virtual environment
        C_ht = lpfet.Householder_orbitals(RDM_KS,N_mo_cl)
        # Compute the 1- and 2-body integrals
        h_Ht = C_ht.T@h_KS_permuted@C_ht
        g_Ht = lpfet.u_matrix(N_mo, U, delocalized_rep=True, orb_coeffs=C_ht)
        # Use the Frozen-core (active space) approximation.
        cluster_indices = [ i for i in range(N_mo_cl) ]
        env_occ_indices = [ N_mo_cl + i for i in range(N_occ_env) ]
        g_Ht_NIB = np.zeros(g_Ht.shape)
        g_Ht_NIB[0,0,0,0] = g_Ht[0,0,0,0]
        core_energy, h_cl_core, g_cl_core = tools.qc_get_active_space_integrals(h_Ht, g_Ht_NIB, env_occ_indices, cluster_indices)
        # Build the Hamiltonian of the cluster using the active space and frozen-core orbitals
        H_cl = tools.build_hamiltonian_quantum_chemistry( h_cl_core, g_cl_core, basis_cl, a_dag_a_cl )
        E_cl, Psi_cl = scipy.linalg.eigh(H_cl.toarray())
        RDM1_cl = tools.build_1rdm_alpha(Psi_cl[:,0], a_dag_a_cl)
        occ_cluster_test[impurity_index] = RDM1_cl[0,0]
        RDM1_cl_free, RDM2_cl_free = tools.build_1rdm_and_2rdm_spin_free( Psi_cl[:,0], a_dag_a_cl )

        E_fragment = np.einsum('q,q', (h_cl_core[0,:]), RDM1_cl_free[0,:])+(1./2)*np.einsum('qrs,qrs', g_cl_core[0,:,:,:], RDM2_cl_free[0,:,:,:])
       
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
    plt.savefig("./figures/density_U{:.3f}_ring_{}_{}_NIB.pdf".format(U_list[i],opt_method,init), format="pdf", bbox_inches="tight")
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
plt.savefig("./figures/Hubbard_ring_energy_{}_{}_NIB.pdf".format(U_list[i],opt_method,init), format="pdf", bbox_inches="tight")
plt.show()
