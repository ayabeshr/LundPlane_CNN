
# coding: utf-8

# # NN Preprocessing Notebook - Jet Images 
# 
# Notebook for pre-proccessing data for neural networks. Reads data from root files and saves it as compressed numpy `.npz` files.

# In[1]:

from __future__ import print_function
from ROOT import gROOT, TFile, TTree, TCanvas, TList, TH2D, TH1D
import numpy as np
import sys, os
import root_numpy as rtnp

import matplotlib.pyplot as plt


# ### Configuration

# In[2]:

#data_type = os.environ["JETTYPE"] #Get bash arguments when running in batch mode
data_type = "QCD" #Set manually in interactive mode

#pt_low = int( os.environ["PTLOW"] )
#pt_high = int( os.environ["PTHIGH"] )


#events_per_tree = int( os.environ["MAXEVENTS"] ) #Run over all events in batch mode

# events_per_tree = -1 #Run on all events in interactive mode

events_per_tree = 10000 #Run only a few events when testig in interactive mode


# In[3]:

#Configuration
gROOT.SetBatch(True)

#Number of bins in x and y
n_bins = 40

#jet image ranges
ji_x_range = [-1, 1]
ji_y_range = [-1, 1]
#lund plane image ranges
lp_x_range = [0, 5]
lp_y_range = [0, 6.5]

# define jet pt range
GeV = 1 #jets stored in GeV for new data in the loc: wprime_20190204/user.asopio.wprime_c...
pt_low = 300 *GeV
pt_high = 400 *GeV

#state jet(s) to analyse
# data_type = "QCD"
# data_type = "W"
opt_input_collection = "track_jets" # "track_jets" or "truth_jets"
jet_radius = 1.0


if opt_input_collection == "truth_jets":
    branches = ["const_eta","const_phi","const_energy","jet_eta","jet_phi","jet_pt","isW"]
if opt_input_collection == "track_jets":
    branches = ["track_eta","track_phi","track_e","rjetEta","rjetPhi","rjetPt","tjetIsW_track"]
    
    
print( "--- Jet image configuration --- " )
print( "Jet collection:", opt_input_collection )
print( "Jet pt range:", pt_low, "-", pt_high, "GeV" )


# ### Jet image creation

# In[ ]:

def make_jet_image( jet_eta, jet_phi, const_etas, const_phis, const_es,
                   eta_binning=[50, -1, 1], phi_binning=[50, -1, 1] ):
    """
    Function to make eta-phi images of jets. 
    
    Arguments:
    - jet_eta: float, eta of the jet
    - jet_phi: float, phi of the jet
    - const_etas: iterable, eta of jet constituents
    - const_phis: iterable, phi of jet constituents
    - const_es: iterable, energy of jet constituents
    - eta_binnning: [int, float, float], number of bins and axis range for eta
    - phi_binnning: [int, float, float], number of bins and axis range for phi
    
    Returns:
    - jet_image: a 2d numpy array.
    """
        
    #Find the maximum energy constituent
    max_energy = np.max(const_es)
    i_max = const_es.index(max_energy)

    # for the cluster with the largest energy only (leading jet)
    
    # set clust_{N,P,E} to leading jet values
    const_N = const_eta[i_max]
    const_P = const_phi[i_max]
    const_E = const_energy[i_max]

    
    # Translate maximum energy constituent

    # // Shift to center at jet (0,0)?
    # //Leading jet - average
    const_N = const_N - jet_eta
    const_P = const_P - jet_phi

    # // correct bounds for overshift    (azimuthal periodicity)
    if const_P > np.pi:
        const_P = const_P - 2*np.pi
    elif const_P < -np.pi:
        const_P = const_P + 2*np.pi
    else:
        const_P = const_P

        
    #Rotate maximum energy constituent
    
    #find angle between cluster and the y-axis 
    const_dist = np.sqrt( (const_N-0)**2 + (const_P-0)**2 )

    theta = 0.0
    # // correct for global theta
    # // quadrant 1
    if const_N > 0 and const_P > 0:
        theta = np.arcsin(const_N/const_dist) #sin^-1 (eta/dist_from_y)
        # print("theta #1:", theta)

    # // quadrant 2
    elif const_N < 0 and const_P > 0:
        const_N = abs(const_N)
        theta = np.arcsin(const_N/const_dist)
        theta = 2*np.pi - theta;
        # print("theta #2:", theta)

    # // quadrant 3
    elif const_N < 0 and const_P < 0:
        const_N = abs(const_N)
        theta = np.arcsin(const_N/const_dist)
        theta = np.pi + theta
        # print("theta #3:", theta)

    # // quadrant 4
    elif const_N > 0 and const_P < 0:
        theta = np.arcsin(const_N/const_dist)
        theta = np.pi - theta
        # print("theta #4:", theta)

    else:
        print("No Quadrant Selected")

    # // rotate this cluster to the y-axis
    const_N = 0;
    const_P = const_dist;

    #rotated constituent 
    const_Nfin = const_N*np.cos(theta) - const_P*np.sin(theta)
    const_Pfin = const_N*np.sin(theta) + const_P*np.cos(theta)

    
    #lists for rotated jet constituents
    const_Nfin_array = []
    const_Pfin_array = []
    const_Efin_array = []
    

    const_Nfin_array.append(const_Nfin)
    const_Pfin_array.append(const_Pfin)
    
    #invariant E
    const_Efin = const_E/np.cosh(const_Nfin)
    const_Efin_array.append(const_E)
                              

    # run through each cluster for all clusters in a jet
    for j in range(len(const_eta)):
        # rotate other clusters by same theta
        if const_energy[j] != max:

            # // get initial cluster
            const_N = const_eta[j]
            const_P = const_phi[j]
            const_E = const_energy[j]

            #const_theta = 2*np.arctan(np.exp(-const_N)) #//why?
            #const_et = abs(const_E*np.cos(const_theta)) #//    transverse energy, var 5 (use in report)

            # // first shift to centralize at jet (0,0)
            const_N = const_N - jet_eta
            const_P = const_P - jet_phi

            # // correct bounds for overshift
            if const_P > np.pi:
                const_P = const_P - 2*np.pi

            elif const_P < -np.pi:
                const_P = const_P + 2*np.pi


            # // rotate the clusters by theta (of highest energy jet)
            const_Nfin = (const_N*np.cos(theta) - const_P*np.sin(theta))
            const_Pfin = (const_N*np.sin(theta) + const_P*np.cos(theta))
            #const_Nfin = const_N #use for HighPt
            #const_Pfin = const_P #use for HighPt

            #invariant E
            const_Efin = const_E/np.cosh(const_Nfin)

            
            const_Nfin_array.append(const_Nfin)
            const_Pfin_array.append(const_Pfin)
            
            const_Efin_array.append(const_Efin)
            
    
    n_bins_eta = eta_binning[0]
    eta_range = eta_binning[1:]
    n_bins_phi = eta_binning[0]
    phi_range = phi_binning[1:]
    
    jet_image, xarr, yarr = np.histogram2d(const_Nfin_array, const_Pfin_array, weights=const_Efin_array,
                                           bins=[n_bins_eta, n_bins_phi], range=[eta_range, phi_range])
    
    return jet_image


# ### Lund Plane creation

# In[ ]:

-


# In[ ]:

h_pt = TH1D("pt_dist", "pt_dist", 100, 0, 6000) 


# ### Loop over all entries in the tree

# In[ ]:

-


# ### Plot jet images

# In[ ]:

from ROOT import gStyle, kLightTemperature, kBlackBody, kCool, kColorPrintableOnGrey , kPastel , kViridis
from IPython.display import Image
print("Number of jet images: ", len(jet_images))
    
sum_images = np.sum(jet_images, axis=0)

h_title = "Jet Image, "+str(pt_low)+" < p_{T} < "+str(pt_high)+" GeV"
h = TH2D( "h_jet_image", h_title, n_bins, ji_x_range[0], ji_x_range[1], n_bins, ji_y_range[0], ji_y_range[1])

rtnp.array2hist(sum_images, h)

#Plot using root
gStyle.SetOptStat(0)
# gStyle.SetPalette(kPastel)
# gStyle.SetPalette(kCool)
gStyle.SetPalette(kColorPrintableOnGrey)

c = TCanvas("c", "c", 800, 600)
c.SetLogz()

#h.SetAxisRange(1, 100000, "Z")
h.GetXaxis().SetTitle("#eta")
h.GetYaxis().SetTitle("#phi")
h.Draw("colz")

c.Print("plots/{}_pt{}_{}_jet_image.png".format(data_type, pt_low, pt_high))
Image("plots/{}_pt{}_{}_jet_image.png".format(data_type, pt_low, pt_high))

#Plot using matplotlib
# plt.imshow(sum_images)


# ### Plot Lund Plane

# In[ ]:

from ROOT import gStyle
from IPython.display import Image
print("Number of Prim Lund Plane images:", len(prim_lund_images))

sum_prim_lund_images = np.sum(prim_lund_images, axis=0)

h1_title = "Prim Lund Plane, "+str(pt_low)+" < p_{T} < "+str(pt_high)+" GeV"
h1 = TH2D( "h1_prim_lund_plane", h1_title, n_bins, lp_x_range[0], lp_x_range[1], n_bins, lp_y_range[0], lp_y_range[1])

#PLot using root
rtnp.array2hist(sum_prim_lund_images, h1)

gStyle.SetOptStat(0)

c = TCanvas("c", "c", 800, 700)

# h.SetAxisRange(0, 700, "Z")
h1.GetXaxis().SetTitle("ln(R/#DeltaR)")
h1.GetYaxis().SetTitle("ln(1/k_{T})")
h1.Draw("colz")


c.SetLogz()
c.Print("plots/{}_pt{}_{}_prim_lund_plane_image.png".format(data_type, pt_low, pt_high))
Image("plots/{}_pt{}_{}_prim_lund_plane_image.png".format(data_type, pt_low, pt_high))

#Plot using matplotlib
# plt.imshow(sum_images)


# In[ ]:

#from ROOT import gStyle
#from IPython.display import Image
#print("Number of Sec Lund Plane images:", len(sec_lund_images))

#sum_sec_lund_images = np.sum(sec_lund_images, axis=0) 

#h2_title = "Sec Lund Plane, "+str(pt_low)+" < p_{T} < "+str(pt_high)+" GeV"
#h2 = TH2D( "h2_sec_lund_plane", h2_title, n_bins, lp_x_range[0], lp_x_range[1], n_bins, lp_y_range[0], lp_y_range[1])

#PLot using root
#rtnp.array2hist(sum_sec_lund_images, h2)

#gStyle.SetOptStat(0)

#c = TCanvas("c", "c", 800, 700)

# h.SetAxisRange(0, 700, "Z")
#h2.GetXaxis().SetTitle("ln(R/#DeltaR)")
#h2.GetYaxis().SetTitle("ln(1/k_{T})")
#h2.Draw("colz")


#c.SetLogz()
#c.Print("plots/{}_pt{}_{}_sec_lund_plane_image.png".format(data_type, pt_low, pt_high))
#Image("plots/{}_pt{}_{}_sec_lund_plane_image.png".format(data_type, pt_low, pt_high))

#Plot using matplotlib
# plt.imshow(sum_images)


# In[ ]:

#from ROOT import gStyle
#from IPython.display import Image
#print("Number of Lund Plane images:", len(lund_images))

#sum_lund_images = np.sum(lund_images, axis=0)

#h_title = "Lund Plane, "+str(pt_low)+" < p_{T} < "+str(pt_high)+" GeV"
#h = TH2D( "h_lund_plane", h_title, n_bins, lp_x_range[0], lp_x_range[1], n_bins, lp_y_range[0], lp_y_range[1])

#PLot using root
#rtnp.array2hist(sum_lund_images, h)

#gStyle.SetOptStat(0)

#c = TCanvas("c", "c", 800, 700)

# h.SetAxisRange(0, 700, "Z")
#h.GetXaxis().SetTitle("ln(R/#DeltaR)")
#h.GetYaxis().SetTitle("ln(1/k_{T})")
#h.Draw("colz")


#c.SetLogz()
#c.Print("plots/{}_pt{}_{}_lund_plane_image.png".format(data_type, pt_low, pt_high))
#Image("plots/{}_pt{}_{}_lund_plane_image.png".format(data_type, pt_low, pt_high))

#Plot using matplotlib
# plt.imshow(sum_images)


# In[ ]:

from ROOT import gStyle
from IPython.display import Image
#print("Number of histogram images:", len(h_images))


c = TCanvas("c", "c", 800, 600)

h_pt.GetXaxis().SetTitle("p_{T}")
h_pt.GetYaxis().SetTitle("Number of jets")
h_pt.Draw("hist")

c.Print("plots/{}_pt{}_{}_pt_dist_image.png".format(data_type, pt_low, pt_high))
Image("plots/{}_pt{}_{}_pt_dist_image.png".format(data_type, pt_low, pt_high))


# ### Save jet images

# In[ ]:

import datetime

outdir = "/mnt/storage/abeshr/"

timestamp = datetime.datetime.now().strftime("%Y%m%d")

data_dict = {
    data_type+"_jet_images": jet_images,
    data_type+"_prim_lund_planes": prim_lund_images, 
    #data_type+"_sec_lund_planes": sec_lund_images,
    #data_type+"_lund_planes": lund_images,
}


outpath = outdir+"{}_{}_images_pt{}_{}_{}bins.npz".format(timestamp, data_type, pt_low, pt_high, n_bins)
np.savez_compressed(outpath, **data_dict)


# In[ ]:




# In[ ]:



