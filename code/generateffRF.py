"""
==================================================
Script Name: generateffRF.py
==================================================

Code Description:
-----------------
This is the code for the generation of the receptive fields
of LM neurons, based on the arrangement and orientation 
selectivity of the synapses

Developers:
-----------
- Nikos Malakasis (nikos.malakasis@tum.de)
- Xinyun Zhang (xy.zhang@tum.de)

Inputs:
--------
- "setting" / "-G" -> "GR45", "GR135" or "CTRL": Select goggle experiment simulation type. Type = string.
- "trials" / "-T" -> <number of trials>: Select the amount of simulations to run. Type = integer.
- "seed" / "-S" -> <seed number>: Set the seed to control randomness. Type = integer.
- "folder" / "-F" -> "generated" or "provided"

Input files:
------------
-"trial<k>_ori.npy" -> Orientations of established synapses at the end of the simulation
-"trial<k>_coord.npy" -> Coordinates of established synapses at the end of the simulation
-"trial<k>_weight.npy" -> Weights of established synapses at the end of the simulation

Outputs:
--------
-"trial<k>_rf.npy" -> The receptive field of the each neuron after the simulation.
-"trial<k>_generated_rf.png' -> Plot showing the receptive field of the each neuron after the simulation.

Dependencies:
-------------
numpy, scipy, gwp, tqdm

Usage:
------
python generateffRF.py -G <setting> -T <trials> -S <seed> -F <folder>

==================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
import gwp
from tqdm import tqdm
from argparse import ArgumentParser


def stim_to_all_inputs(stim, oridx_arr, coord_arr):
  # Define Gabor filter banks parameters

  N = 80
  num_ori = 8
  sigma = 3
  frequency = 0.5 / sigma

  # configure gabor filter kernals for V1 and LM processing
  orientations = gwp.n2orientations(8)
  bank_v1 = [gwp.gaborbank(sigma=2*sigma, orientations=orientations, cyclespersigma = 0.5, nsigma=1, phase=thisphase)
                  for thisphase in [0, np.pi/2]]

  all_convolved = []
  all_syn_inputs = np.zeros(oridx_arr.shape[0])
  offset = bank_v1[0][:,:,0,0].shape[0]
  for ii in range(num_ori):
    convolved_img = np.square(signal.convolve2d(stim, bank_v1[0][:,:,0,ii])) + np.square(signal.convolve2d(stim, bank_v1[1][:,:,0,ii]))
    all_convolved.append(convolved_img)
  for synii in range(N):
    coord = coord_arr[synii]
    oridx = oridx_arr[synii]
    coord_x = int(coord[0] * stim.shape[0])
    coord_y = int(coord[1] * stim.shape[0])
    this_syn_input = np.mean(all_convolved[oridx][coord_x : coord_x+offset, coord_y : coord_y+offset])
    all_syn_inputs[synii] = this_syn_input
  return all_syn_inputs


parser=ArgumentParser()
parser.add_argument('-G','--setting',type=str)
parser.add_argument('-T','--trials',type=int)
parser.add_argument('-S','--seed',type=int)
parser.add_argument('-F','--folder',type=str)

args = vars(parser.parse_args())

allseed=args["seed"]
np.random.seed(allseed)

GR45 = False
GR135 = False
Control = False

if args["setting"]=="GR45":
  GR45=True

if args["setting"]=="GR135":
  GR135=True

if args["setting"]=="CTRL":
  Control=True


# Define Gabor filter banks parameters
sigma = 3
frequency = 0.5 / sigma

# configure gabor filter kernals for V1 and LM processing
orientations = gwp.n2orientations(8)
bank_v1 = [gwp.gaborbank(sigma=2*sigma, orientations=orientations, cyclespersigma = 0.5, nsigma=1, phase=thisphase)
                for thisphase in [0, np.pi/2]]

# Initial weight value
############################################################################
# The method for computing receptive field shapes                          #
# by exciting the feedforward model with small visual stimuli              #
# of different orientations, at different locations across visual field    #
#                                                                          #
# Results already exist in the data folder                                 #
# Replace dummy 'data_directory' with path name                            #
############################################################################

setting_folder=""

if Control:
  setting_folder="CTRL/"
if GR45:
  setting_folder="GR45/"
if GR135:
  setting_folder="GR135/"

if args["folder"]=="generated":
  file_oh_load = '../generated_model_outputs/FF_model_outputs/'+setting_folder # path directory to save data
  file_oh_save = '../generated_model_outputs/FF_model_outputs/'+setting_folder # path directory to save data

if args["folder"]=="provided":
  file_oh_load = '../provided_model_outputs/FF_model_outputs/feedforward_results/'+setting_folder # path directory to save data
  file_oh_save = '../provided_model_outputs/FF_model_outputs/new_outputs/'+setting_folder # path directory to save data



all_test_stim = []
wid_visualstim = bank_v1[0][:,:,0,0].shape[0]

num_trials = args["trials"]

for k in range(0,num_trials):
  trial_ori = np.load(file_oh_load +'trial'+ str(k) + '_ori.npy')
  trial_coord = np.load(file_oh_load +'trial'+ str(k) + '_coord.npy')
  trial_weight = np.load(file_oh_load +'trial'+ str(k) + '_weight.npy')
  rf_map_bias = np.zeros((30,30))
  for test_ori in tqdm(range(8),desc="Receptive Field " + str(k) + " Generation"):
    for ii in range(0,91-wid_visualstim,3):
      for jj in range(0,91-wid_visualstim,3):
        test_image = np.zeros((91,91))
        test_image[ii:ii+wid_visualstim,jj:jj+wid_visualstim] = bank_v1[0][:,:,0,test_ori]
        all_test_stim.append(test_image)
        all_V1_resp = stim_to_all_inputs(test_image, trial_ori[:].astype(np.int64), trial_coord[:,:])
        rf_map_bias[int(ii/3),int(jj/3)] += np.dot(all_V1_resp, trial_weight[:])
   
  np.save(file_oh_save+'trial' + str(k)+'_rf.npy',rf_map_bias)
  plt.figure()
  plt.title('Trial '+str(k)+' receptive field')
  plt.yticks([])
  plt.xticks([])
  plt.imshow(rf_map_bias)
  plt.savefig(file_oh_save+'trial'+str(k)+'_generated_rf.png')
