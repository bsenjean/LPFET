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

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

file_name = ["DET","DET_without_global","DET_KScluster","LPFET","LPFET_KScluster"]
list_R = []
list_E = []
list_E_FCI = []
for name in file_name:
  list_R += [[]]
  list_E += [[]]
  list_E_FCI += [[]]

for i in range(len(file_name)):
  path_name = "../results/Hchain_{}.out".format(file_name[i])
  with open(path_name,"r") as f:
     next(f)
     for line in f:
      values = [s for s in line.split()]
      list_R[i] += [float(values[0])]
      list_E_FCI[i] += [float(values[1])]
      list_E[i] += [float(values[2])]

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(15, 12),sharex=True)

ax.grid(alpha=0.2)
ax.set_ylabel('Energy [hartree]', size=45)
ax.set_xlabel('R (angstrom)', size=45)
ax.plot( np.divide(np.asarray(list_R[0]),np.asarray(list_R[0]) + 4), list_E_FCI[0], color="black", ls='-', label="FCI")
for i in range(len(file_name)):
  ax.plot( np.divide(np.asarray(list_R[i]),np.asarray(list_R[i]) + 4), list_E[i], color=colorFader("red","blue",i/len(file_name)), ls='--', marker=i+2, label=file_name[i], markersize=15)
ax.legend(framealpha=0.8, prop={'size': 20}, loc="best")
ax.set_ylim([-3.5,-1])
plt.tight_layout()
plt.savefig('energy_Hchain.pdf'.format(name))
