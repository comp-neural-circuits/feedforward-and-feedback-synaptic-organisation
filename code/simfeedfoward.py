"""
==================================================
Script Name: simfeedforward.py
==================================================

Code Description:
-----------------
This is the code for the simulation of the feedforward model,
where orientation selective feedforward inputs from the 
primary visual cortex (V1) shape the receptive field of neurons 
in the latero-medial visual cortex (LM).

Developers:
-----------
- Nikos Malakasis (nikos.malakasis@tum.de)
- Xinyun Zhang (xy.zhang@tum.de)

Inputs:
--------
- "setting" / "-G" -> "GR45", "GR135" or "CTRL": Select goggle experiment simulation type. Type = string.
- "trials" / "-T" -> <number of trials>: Select the amount of simulations to run. Type = integer.
- "seed" / "-S" -> <seed number>: Set the seed to control randomness. Type = integer.

Outputs:
--------
-"trial<k>_ori.npy" -> Orientations of established synapses at the end of the simulation
-"trial<k>_coord.npy" -> Coordinates of established synapses at the end of the simulation
-"trial<k>_weight.npy" -> Weights of established synapses at the end of the simulation
-"trial<k>_ori_t0.npy" -> Orientations of established synapses at the start of the simulation
-"trial<k>_coord_t0.npy" -> Coordinates of established synapses at the start of the simulation
-"trial<k>__syns_in_vis_space.png" -> Plot of final distribution of established synapses in visual space

Dependencies:
-------------
Numpy, Scipy, tqdm, gwp, matplotlib

Usage:
------
python simfeedforward.py -G <setting> -T <trials> -S <seed>

==================================================
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
from scipy import signal
from tqdm import tqdm
import gwp
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# functions to generate V1 synaptic input
def stim_to_syn_input(stim, oridx, coord):
    # Define Gabor filter banks parameters
  sigma = 3
  # configure gabor filter kernals for V1 processing
  orientations = gwp.n2orientations(8)
  bank_v1 = [gwp.gaborbank(sigma=2*sigma, orientations=orientations, cyclespersigma = 0.5, nsigma=1, phase=thisphase)
                  for thisphase in [0, np.pi/2]]

  convolved_img = np.square(signal.convolve2d(stim, bank_v1[0][:,:,0,oridx])) + np.square(signal.convolve2d(stim, bank_v1[1][:,:,0,oridx]))
  offset = bank_v1[0][:,:,0,oridx].shape[0]
  coord_x = int(coord[0] * stim.shape[0])
  coord_y = int(coord[1] * stim.shape[0])
  response = np.mean(convolved_img[coord_x : coord_x+offset, coord_y : coord_y+offset])
  return response

def stim_to_all_inputs(stim, oridx_arr, coord_arr):
  N=80
  num_ori=8
  # Define Gabor filter banks parameters
  sigma = 3
  # configure gabor filter kernals for V1 processing
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

def new_syn(soma_coord, dist_sigma):
  soma_coordx = soma_coord[0]
  soma_coordy = soma_coord[1]
  new_syn_coord = np.random.rand(1,2) * 0.6 + 0.2
  new_syn_coord[0][0] = np.random.normal(soma_coordx, dist_sigma/2)
  new_syn_coord[0][1] = np.random.normal(soma_coordy, dist_sigma)
  new_syn_coord = np.clip(new_syn_coord, 0,1)
  return new_syn_coord

############################################################################
######################## Main simulation function ##########################
############################################################################

def simulate_moving_stim(soma_coordx, soma_coordy):
  for tt in tqdm(range(1,int(T)),desc="Main Simulation"):
    # Every 20*50ms: update input stimulus image
    if not np.mod(tt, t_frame):
      curr_stim = all_stim[int(tt/t_frame)]
      # Pre-syn: V1
      curr_all_pre = stim_to_all_inputs(curr_stim, syn_orientations, syn_coordinates)
      for ii in range(L):
        S[ii,tt:tt+t_frame] = np.random.rand(1,t_frame) < curr_all_pre[ii]/20
    Sin = S[:,tt]
    R[: , tt] = R[: , tt-1] * np.exp(-1./tauR) + phi*Sin*(1 - np.exp(-1./tauR))
    Waug = np.tile(W[: , tt-1], (L, 1))*SMat
    Y[: , tt] = Y[: , tt-1]*np.exp(-1./tauY) + (Waug@Sin)*(1 - np.exp(-1./tauY))
    # Hebbian excitatory plasticity in FeedForward model
    W[: , tt] =  np.clip(W[: , tt-1] + (1./tauW)*( (np.clip(Y[: , tt-1],0,None))*(R[: , tt-1] + rho) ) , 0, 1)

    idx=0
    for synw in W[:, tt]:
      if synw < 0.02 and mask[idx]==True:
        dedSyn=idx+dist+1
        newSyn=np.random.choice(AllSlots)
        #Replace chosen position in available slots with the position left open by the ded synapse
        AllSlots[AllSlots==newSyn]=dedSyn
        AllSlots.sort()
        # new pre retinotopic coordinates and orientation
        newpre_ori = np.random.choice(np.arange(0, 8), p=p_ori) # biased distribution
        newpre_coord = new_syn([soma_coordx, soma_coordy], 0.1)
        syn_orientations[idx] = newpre_ori
        syn_coordinates[idx,:] = newpre_coord
        # create new input spike train at new synapse
        newpre_rate = stim_to_syn_input(curr_stim, newpre_ori, newpre_coord[0])
        # transfer input spiketrain to new synapse
        win_rest = t_frame - np.mod(tt,t_frame)
        S[newSyn-dist-1,tt:tt+win_rest]= np.random.rand(1,win_rest) < newpre_rate/20
        S[dedSyn-dist-1,tt:]=0
        # mask ded syn and unmask new syn
        mask[dedSyn-dist-1]=False
        mask[newSyn-dist-1]=True
        # trace new indices to sort vectors
        W[dedSyn-dist-1,tt]=0
        W[newSyn-dist-1,tt]=Winit
      idx+=1
    idx_temp = idx
    all_syn_ori[:,tt] = syn_orientations
    all_syn_coord[:,tt,:] = syn_coordinates

  return W, all_syn_ori, all_syn_coord


parser=ArgumentParser()
parser.add_argument('-G','--setting',type=str)
parser.add_argument('-T','--trials',type=int)
parser.add_argument('-S','--seed',type=int)

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

# Initial weight value
Winit = 0.2
# rho variable controls the amount of depression
# when there is no presynaptic activity
eta = 0.45
rho = -0.5
# efficiency constant
phi = 3
normSTD = 6000
XDUR = 50
# weight time constant
tauw = 6000
tauW = tauw/XDUR * (1/(2*(1 - eta)))
# postsynaptic accumulator time constant
tauy = 300
tauY = tauy/XDUR
# presynaptic accumulator time constant
taum = 100
tauR = taum/XDUR
dist=150
# Total possible synapses locations on dendrite
L = 100
# Number of total synapses on dendrite
N = 80

# 8 discrete orientations, every 22.5 deg, to sample from
num_ori = 8

#Choose initial positions
AllSlots=np.arange(1,L+1)+dist
AllPos=AllSlots.reshape(-1,1)
randpos= np.array(np.random.choice(AllSlots,N,replace=False)).reshape(-1,1)
initidc = randpos.argsort(axis=0)
pos=randpos[initidc].reshape(randpos.shape)
for i in pos:
	AllSlots=np.delete(AllSlots,np.where(AllSlots==i))

# Distance between all dendrite locations
dMat = cdist(AllPos,AllPos)

#This matrix determines the calcium spread postsynaptically
SMat = norm.pdf(dMat , 0 , normSTD)*(np.sqrt(2*np.pi)*normSTD)

# Stimulus image generation: Mooving bars with different orientations
all_stim = []
image = np.zeros((181,91))
bar_width = 8
fade_in_range = bar_width // 2
start_row = (image.shape[0] - bar_width) // 2
end_row = start_row + bar_width
for i in range(start_row, end_row):
    fade_value = (i - start_row) / fade_in_range if i - start_row < fade_in_range else (end_row - i) / fade_in_range
    image[i, :] = fade_value
image_90 = np.rot90(image)

height, width = 91, 183
image_135 = np.zeros((height, width), dtype=np.uint8)
thickness = 3
center_x = width // 2
center_y = height // 2
start_x = center_x - min(center_y, center_x)
start_y = center_y - min(center_y, center_x)
end_x = center_x + min(center_y, width - center_x - 1)
end_y = center_y + min(height - center_y - 1, center_x)
for t in range(-thickness // 2, thickness // 2 + 1):
    for i in range(end_x - start_x + 1):
        x = start_x + i
        y = start_y + i + t
        if 0 <= x < width and 0 <= y < height:
            image_135[y, x] = 1
image_45 = np.rot90(image_135)

# Create a pool of stimuli at different orientations and locations
# for assembling moving stimuli during simulation
# 0 - 0 deg - Horizontal
# 1 - GR135 overrepresented orientation
# 2 - 90 deg - Vertical
# 3 - GR45 overrepresented orientation

all_stim_pool = [[],[],[],[]]
all_stim = []
for ii in range(0,183-91,3):
  all_stim_pool[0].append(image[ii:ii+91, :])
for ii in range(0,183-91,3):
  all_stim_pool[1].append(image_45[ii:ii+91, :])
for ii in range(0,183-91,3):
  all_stim_pool[2].append(image_90[:, ii:ii+91])
for ii in range(0,183-91,3):
  all_stim_pool[3].append(image_135[:, ii:ii+91])


# Presynaptic inputs orientation selectivity probability distributions
if Control:
  p_ori = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
# GR45 biased orientation probability distribution
if GR45:
  p_ori = [0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.125, 0.4375, 0.125]
# GR135 biased orientation probability distribution
if GR135:
  p_ori = [0.0625, 0.125, 0.4375, 0.125, 0.0625, 0.0625, 0.0625, 0.0625]



##################################
# Run simulations and save data  #
##################################

setting_folder=""

if Control:
  setting_folder="CTRL/"
if GR45:
  setting_folder="GR45/"
if GR135:
  setting_folder="GR135/"


file_oh = '../generated_model_outputs/FF_model_outputs/'+setting_folder # path directory to save data


soma_coordx = 0.5
soma_coordy = 0.5

if Control:
  stim_seq = [0,1,2,3]*4
if GR45:
  stim_seq = [0,1,2,3]*2 +[3]*8
if GR135:
  stim_seq = [0,1,2,3]*2 +[1]*8

t_frame = 5

num_trials = args["trials"]
colors_rgb=[(200, 125, 33),(245, 118, 33),(255, 0, 196),(222, 126, 110),(0, 168, 161),(42, 79, 186),(112, 0, 159),(163, 86, 153)]
colors=np.array([(r/255, g/255, b/255) for (r, g, b) in colors_rgb])

for k in range(0,num_trials):
  # Recreate randomized sequence of stimuli for each simulation
  # which follows the described distribution
  np.random.shuffle(stim_seq)
  all_stim = []
  for x in range(len(stim_seq)):
    curr_img = all_stim_pool[stim_seq[x]]
    for ii in range(0,183-91,3):
      all_stim.append(all_stim_pool[stim_seq[x]][int(ii/3)])
    for ii in range(183-91,0,-3):
      all_stim.append(all_stim_pool[stim_seq[x]][int(ii/3)])
  T = len(all_stim) * t_frame
  # Re-initialize simulation matrix before each trial
  R = np.zeros((L,T))
  Y = np.zeros((L,T))
  W = np.zeros((L,T))
  S = np.zeros((L,T))
  mask=np.zeros(L)
  mask[pos-dist-1]=1
  mask=mask.astype(bool)
  Ysoma=np.zeros((1,T))
  # Re-initialize synapse properties: coordinate, orientation, weight
  W[:,0][mask] = Winit
  all_pre = np.zeros((L,T))
  soma_rate = np.zeros((1,T))
  all_syn_ori = np.zeros((L,T))
  all_syn_coord = np.zeros((L,T,2))
  # RF centers for all feedforward synapses
  # retinotopic Gaussian distribution with horizontally elongated shape
  syn_coordinates = np.random.rand(L, 2)
  for ii in range(L):
    syn_coordinates[ii,0] = np.random.normal(0.5, 0.1/2)
    syn_coordinates[ii,1] = np.random.normal(0.5, 0.1)
  syn_orientations = np.random.choice(np.arange(0, 8), size = L, p=p_ori)
  all_syn_ori[:,0] = syn_orientations
  all_syn_coord[:,0,:] = syn_coordinates

  # Initial stimulus
  curr_stim = all_stim[0]
  curr_all_pre = stim_to_all_inputs(curr_stim, syn_orientations, syn_coordinates)
  for ii in range(L):
    S[ii,0:100] = np.random.rand(1,100) < curr_all_pre[ii]/20

  # Present all stimuli to the model and track changes at all synapses
  W, all_syn_ori, all_syn_coord = simulate_moving_stim(soma_coordx, soma_coordy)
  curr_t = T-1

  np.save(file_oh + '/trial'+str(k)+'_ori.npy',all_syn_ori[:,curr_t])
  np.save(file_oh + '/trial'+str(k)+'_coord.npy',all_syn_coord[:,curr_t,:])
  np.save(file_oh + '/trial'+str(k)+'_weight.npy',W[:,curr_t])
  np.save(file_oh + '/trial'+str(k)+'_ori_t0.npy',all_syn_ori[:,0])
  np.save(file_oh + '/trial'+str(k)+'_coord_t0.npy',all_syn_coord[:,0,:])

  #Generate plot of final synapses in visual space

  plt.figure(figsize=(6,6))

  plt.title('Trial '+str(k)+' final synapse allocation in visual space')
  plt.scatter(all_syn_coord[:,curr_t,1], 1-all_syn_coord[:,curr_t,0], c=colors[np.array(all_syn_ori[:,curr_t]).astype(int)], s=500*W[:,curr_t])
  plt.xlim([0.1,0.9])
  plt.ylim([0.1,0.9])
  plt.yticks([])
  plt.xticks([])

  plt.savefig(file_oh + '/trial'+str(k)+'_syns_in_vis_space.png')