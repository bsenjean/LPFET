import numpy as np  
import scipy 
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt  
from colour import Color
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import BSpline
from scipy.interpolate import interp1d


# Parameter for sexy plots 
plt.rc('font',  family='serif')
#plt.rc('font',  serif='Times New Roman')
#plt.rc('font',  serif='Helvetica')
plt.rc('font',  size='33') 
plt.rc('xtick', labelsize='Large')
plt.rc('ytick', labelsize='Large')
# plt.rc('legend', fontsize='medium')
plt.rc('lines', linewidth='4')
#plt.rcParams.update({ "text.usetex": True})

ICGMmarine=(0.168,0.168,0.525)
ICGMblue=(0,0.549,0.714)
ICGMorange=(0.968,0.647,0)
ICGMyellow=(1,0.804,0)
yellow2=(1,0.95,0)

file_name = ["DET","DET_without_global","DET_KScluster","LPFET","LPFET_KScluster"]

for name in file_name:
  path_name = "../results/Hubbard_ring_{}.out".format(name)
  list_U = []
  list_E = []
  list_E_FCI = []
  with open(path_name,"r") as f:
     next(f)
     for line in f:
      values = [s for s in line.split()]
      list_U += [float(values[0])]
      list_E_FCI += [float(values[1])]
      list_E += [float(values[2])]

  list_U = np.asarray(list_U)
    
  fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(15, 12),sharex=True)
  
  ax.grid(alpha=0.2)
  ax.set_ylabel('Energy/t', size=45)
  ax.set_xlabel('U/(U + 4t)', size=45)
  ax.plot( np.divide(list_U,list_U + 4), list_E_FCI, color="black", ls='-', label="FCI")
  ax.plot( np.divide(list_U,list_U + 4), list_E, color=ICGMblue, ls='--', label="Embedding", marker='o')
  ax.legend(framealpha=0.8, prop={'size': 30}, loc="best")
  plt.tight_layout()
  plt.savefig('energy_Hubbard_ring_{}.pdf'.format(name))
