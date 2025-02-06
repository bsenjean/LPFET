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
list_U = []
list_n_FCI = []
list_n_cluster = []
list_n_KS = []
for name in file_name:
  list_U += [[]]
  list_n_FCI += [[]]
  list_n_cluster += [[]]
  list_n_KS += [[]]

for i in range(len(file_name)):
  path_name = "../results/Hubbard_ring_{}.out".format(file_name[i])
  with open(path_name,"r") as f:
     next(f)
     for line in f:
      values = [s for s in line.split()]
      list_U[i] += [float(values[0])]
      list_n_FCI[i] += [[float(values[3]),float(values[4]),float(values[5]),float(values[6]),float(values[7]),float(values[8])]]
      list_n_cluster[i] += [[float(values[9]),float(values[10]),float(values[11]),float(values[12]),float(values[13]),float(values[14])]]
      list_n_KS[i] += [[float(values[15]),float(values[16]),float(values[17]),float(values[18]),float(values[19]),float(values[20])]]

# assuming the same list_U is used for any file...
for U_index in range(len(list_U[0])):
   fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(15, 12),sharex=True)
   
   ax.grid(alpha=0.2)
   ax.set_ylabel('Orbital occupation', size=45)
   ax.set_xlabel('site', size=45)
   ax.plot(np.array([i for i in range(6)]), list_n_FCI[0][U_index], color="black", ls='-', label="FCI")
   for i in range(len(file_name)):
     ax.plot( np.array([i for i in range(6)]), list_n_cluster[i][U_index], color=colorFader("red","blue",i/len(file_name)), ls='--', marker=i+2, label=file_name[i], markersize=15)
     ax.plot( np.array([i for i in range(6)]), list_n_KS[i][U_index], color=colorFader("red","blue",i/len(file_name)), ls=':', marker=i+2, markersize=15)
   ax.legend(framealpha=0.8, prop={'size': 20}, loc="best")
   plt.tight_layout()
   plt.savefig('density_Hubbard_ring_U{}.pdf'.format(list_U[0][U_index]))
