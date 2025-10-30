"""
==================================================
Script Name: processfbmodelresults.py
==================================================

Code Description:
-----------------
This is the code for the analysis of the results generated 
by the feedback model. Specifically, the ratio of boutons 
in the 135 degree bowtie is calculated for multiple V1 
neurons over all 3 groups and statistically compared 
across the groups. 

Developers:
-----------
- Nikos Malakasis (nikos.malakasis@tum.de)
- Xinyun Zhang (xy.zhang@tum.de)

Inputs:
--------
- "trials" / "-T" -> <number of trials>: Select the amount of simulations to run. 
In this analysis code, total number of trials is distributed between all 4 possible
V1 neuron soma orientations, according to their representation, as observed in the
experiments (See paper Methods!). RUN IN MULTIPLES OF 100. Type = integer.
- "seed" / "-S" -> <seed number>: Set the seed to control randomness. Type = integer.
- "folder" / "-F" -> "generated" or "provided".

Input files:
------------
-"trial<k>_ori.npy" -> Orientations of established synapses at the end of the simulation
-"trial<k>_coord.npy" -> Coordinates of established synapses at the end of the simulation
-"trial<k>_weight.npy" -> Weights of established synapses at the end of the simulation

Outputs:
--------
-"V1_FBRF_LM_bouton_ratio_in_135_deg_bowtie.png" -> Boxplots for the ratio of all boutons in
the 135 degree bowtie of all three groups, statistically compared.
-"V1_FBRF_LM_bouton_ratio_in_135_deg_bowtie_no_res.png" -> Boxplots for the ratio of non orientation responsive
boutons in the 135 degree bowtie of all three groups, statistically compared.
-"V1_FBRF_LM_bouton_ratio_in_135_deg_bowtie_res.png" -> Boxplots for the ratio of orientation responsive
boutons in the 135 degree bowtie of all three groups, statistically compared.
-"V1_FBRF_LM_deltafractions.png" ->Boxplots for the ratio of orientation responsive
 and non orientation responsive boutons in the 135 degree bowtie of all three groups
-"V1_FBRF_LM_bouton_ratio_in_135_deg_bowtie_res_no_res.png" ->Boxplots for the ratio of orientation responsive
 and non orientation responsive boutons in the 135 degree bowtie of all three groups, with differences indicated.

Dependencies:
-------------
numpy, matplotlib, seaborn, pandas, scipy, starbars

Usage:
------
python processfbmodelresults.py -T <trials> -S <seed> -F <folder>

==================================================
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import starbars
from argparse import ArgumentParser


def make_group_palette(palette_like, groups):
    if isinstance(palette_like, dict):
        return {g: palette_like[g] for g in groups}
    try:
        cols = list(sns.color_palette(palette_like, n_colors=len(groups)))
    except Exception:
        cols = list(palette_like)
    if len(cols) < len(groups):
        raise ValueError("Not enough colors for the number of groups.")
    return {g: cols[i] for i, g in enumerate(groups)}


parser=ArgumentParser()
parser.add_argument('-T','--trials',type=int)
parser.add_argument('-S','--seed',type=int)
parser.add_argument('-F','--folder',type=str)

args = vars(parser.parse_args())

allseed=args["seed"]
np.random.seed(allseed)

if args["folder"]=="generated":
  file_oh_load = '../generated_model_outputs/FB_model_outputs/' 
  file_oh_save = '../generated_model_outputs/FB_model_outputs/'

if args["folder"]=="provided":
  file_oh_load = '../provided_model_outputs/FB_model_outputs/feedback_results/' 
  file_oh_save = '../provided_model_outputs/FB_model_outputs/new_outputs/'

#Total number of trials per group.
num_trials=args["trials"]


#############################################
# Process and plot data set from simulation #
#              CONTROL Group                #
#############################################


file_oh_CTRL_0 = file_oh_load + '/FB_CTRL_soma_ori_0/trial'
file_oh_CTRL_2 = file_oh_load + '/FB_CTRL_soma_ori_2/trial'
file_oh_CTRL_4 = file_oh_load + '/FB_CTRL_soma_ori_4/trial'
file_oh_CTRL_6 = file_oh_load + '/FB_CTRL_soma_ori_6/trial'



## soma0 --> soma oridx = 4 --> -- horizontal
count_CTRL_soma0_135_bowtie_ratio = []
count_CTRL_soma0_no_ori_135_bowtie_ratio = []
count_CTRL_soma0_with_ori_135_bowtie_ratio = []
count_CTRL_soma0_deltafraction = []

## soma45 --> soma oridx = 6 --> \
count_CTRL_soma45_135_bowtie_ratio = []
count_CTRL_soma45_no_ori_135_bowtie_ratio = []
count_CTRL_soma45_with_ori_135_bowtie_ratio = []
count_CTRL_soma45_deltafraction = []

## soma90 --> soma oridx = 0 --> | vertical
count_CTRL_soma90_135_bowtie_ratio = []
count_CTRL_soma90_no_ori_135_bowtie_ratio = []
count_CTRL_soma90_with_ori_135_bowtie_ratio = []
count_CTRL_soma90_deltafraction = []

## soma135 --> soma oridx = 2 --> /
count_CTRL_soma135_135_bowtie_ratio = []
count_CTRL_soma135_no_ori_135_bowtie_ratio = []
count_CTRL_soma135_with_ori_135_bowtie_ratio = []
count_CTRL_soma135_deltafraction = []

w_th = 0.05
L=200

##############################################################################################
## soma: 0 deg --> soma oridx = 4 --> -- horizontal

for i in range(int(num_trials/4)):

  trial_ori = np.load(file_oh_CTRL_4 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_CTRL_4 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_CTRL_4 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count_45_bowtie_no_ori = 0
  count_135_bowtie_no_ori = 0
  count_45_bowtie_with_ori = 0
  count_135_bowtie_with_ori = 0

  for ii in range(L):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
      if trial_ori[ii]==-1:
        count_45_bowtie_no_ori += 1
      else:
        count_45_bowtie_with_ori += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
      if trial_ori[ii]==-1:
        count_135_bowtie_no_ori += 1
      else:
        count_135_bowtie_with_ori += 1


  count_CTRL_soma0_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))

  no_ori_135_bowtie=count_135_bowtie_no_ori/(count_135_bowtie_no_ori+count_45_bowtie_no_ori)
  count_CTRL_soma0_no_ori_135_bowtie_ratio.append(no_ori_135_bowtie)

  with_ori_135_bowtie=count_135_bowtie_with_ori/(count_135_bowtie_with_ori+count_45_bowtie_with_ori)
  count_CTRL_soma0_with_ori_135_bowtie_ratio.append(with_ori_135_bowtie)

  deltafraction = with_ori_135_bowtie - no_ori_135_bowtie
  count_CTRL_soma0_deltafraction.append(deltafraction)


##############################################################################################
## soma: 45 deg --> soma oridx = 6 --> \

for i in range(int(num_trials/4)):

  trial_ori = np.load(file_oh_CTRL_6 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_CTRL_6 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_CTRL_6 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count_45_bowtie_no_ori = 0
  count_135_bowtie_no_ori = 0
  count_45_bowtie_with_ori = 0
  count_135_bowtie_with_ori = 0

  for ii in range(L):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
      if trial_ori[ii]==-1:
        count_45_bowtie_no_ori += 1
      else:
        count_45_bowtie_with_ori += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
      if trial_ori[ii]==-1:
        count_135_bowtie_no_ori += 1
      else:
        count_135_bowtie_with_ori += 1

  count_CTRL_soma45_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))

  no_ori_135_bowtie=count_135_bowtie_no_ori/(count_135_bowtie_no_ori+count_45_bowtie_no_ori)
  count_CTRL_soma45_no_ori_135_bowtie_ratio.append(no_ori_135_bowtie)

  with_ori_135_bowtie=count_135_bowtie_with_ori/(count_135_bowtie_with_ori+count_45_bowtie_with_ori)
  count_CTRL_soma45_with_ori_135_bowtie_ratio.append(with_ori_135_bowtie)

  deltafraction = with_ori_135_bowtie - no_ori_135_bowtie
  count_CTRL_soma45_deltafraction.append(deltafraction)


##############################################################################################
## soma: 90 deg --> soma oridx = 0 --> | vertical

for i in range(int(num_trials/4)):

  trial_ori = np.load(file_oh_CTRL_0 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_CTRL_0 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_CTRL_0 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count_45_bowtie_no_ori = 0
  count_135_bowtie_no_ori = 0
  count_45_bowtie_with_ori = 0
  count_135_bowtie_with_ori = 0

  for ii in range(L):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
      if trial_ori[ii]==-1:
        count_45_bowtie_no_ori += 1
      else:
        count_45_bowtie_with_ori += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
      if trial_ori[ii]==-1:
        count_135_bowtie_no_ori += 1
      else:
        count_135_bowtie_with_ori += 1

  count_CTRL_soma90_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))

  no_ori_135_bowtie=count_135_bowtie_no_ori/(count_135_bowtie_no_ori+count_45_bowtie_no_ori)
  count_CTRL_soma90_no_ori_135_bowtie_ratio.append(no_ori_135_bowtie)

  with_ori_135_bowtie=count_135_bowtie_with_ori/(count_135_bowtie_with_ori+count_45_bowtie_with_ori)
  count_CTRL_soma90_with_ori_135_bowtie_ratio.append(with_ori_135_bowtie)

  deltafraction = with_ori_135_bowtie - no_ori_135_bowtie
  count_CTRL_soma90_deltafraction.append(deltafraction)


##############################################################################################
## soma: 135 deg --> soma oridx = 2 --> /

for i in range(int(num_trials/4)):

  trial_ori = np.load(file_oh_CTRL_2 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_CTRL_2 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_CTRL_2 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count_45_bowtie_no_ori = 0
  count_135_bowtie_no_ori = 0
  count_45_bowtie_with_ori = 0
  count_135_bowtie_with_ori = 0

  for ii in range(L):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
      if trial_ori[ii]==-1:
        count_45_bowtie_no_ori += 1
      else:
        count_45_bowtie_with_ori += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
      if trial_ori[ii]==-1:
        count_135_bowtie_no_ori += 1
      else:
        count_135_bowtie_with_ori += 1

  count_CTRL_soma135_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))

  no_ori_135_bowtie=count_135_bowtie_no_ori/(count_135_bowtie_no_ori+count_45_bowtie_no_ori)
  count_CTRL_soma135_no_ori_135_bowtie_ratio.append(no_ori_135_bowtie)

  with_ori_135_bowtie=count_135_bowtie_with_ori/(count_135_bowtie_with_ori+count_45_bowtie_with_ori)
  count_CTRL_soma135_with_ori_135_bowtie_ratio.append(with_ori_135_bowtie)

  deltafraction = with_ori_135_bowtie - no_ori_135_bowtie
  count_CTRL_soma135_deltafraction.append(deltafraction)


#############################################
# Process and plot data set from simulation #
#                 GR45 Group                #
#############################################
  

file_oh_0_GR45 = file_oh_load + '/FB_GR45_soma_ori_0/trial'
file_oh_2_GR45 = file_oh_load + '/FB_GR45_soma_ori_2/trial'
file_oh_4_GR45 = file_oh_load + '/FB_GR45_soma_ori_4/trial'
file_oh_6_GR45 = file_oh_load + '/FB_GR45_soma_ori_6/trial'

## soma0 --> soma oridx = 4 --> -- horizontal
count_GR45_soma0_135_bowtie_ratio = []
count_GR45_soma0_no_ori_135_bowtie_ratio = []
count_GR45_soma0_with_ori_135_bowtie_ratio = []
count_GR45_soma0_deltafraction = []

## soma45 --> soma oridx = 6 --> \
count_GR45_soma45_135_bowtie_ratio = []
count_GR45_soma45_no_ori_135_bowtie_ratio = []
count_GR45_soma45_with_ori_135_bowtie_ratio = []
count_GR45_soma45_deltafraction = []

## soma90 --> soma oridx = 0 --> | vertical
count_GR45_soma90_135_bowtie_ratio = []
count_GR45_soma90_no_ori_135_bowtie_ratio = []
count_GR45_soma90_with_ori_135_bowtie_ratio = []
count_GR45_soma90_deltafraction = []

## soma135 --> soma oridx = 2 --> /
count_GR45_soma135_135_bowtie_ratio = []
count_GR45_soma135_no_ori_135_bowtie_ratio = []
count_GR45_soma135_with_ori_135_bowtie_ratio = []
count_GR45_soma135_deltafraction = []

##############################################################################################
## soma0 --> soma oridx = 4 --> -- horizontal

for i in range(int(num_trials/10)):

  trial_ori = np.load(file_oh_4_GR45 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_4_GR45 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_4_GR45 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count_45_bowtie_no_ori = 0
  count_135_bowtie_no_ori = 0
  count_45_bowtie_with_ori = 0
  count_135_bowtie_with_ori = 0

  for ii in range(L):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
      if trial_ori[ii]==-1:
        count_45_bowtie_no_ori += 1
      else:
        count_45_bowtie_with_ori += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
      if trial_ori[ii]==-1:
        count_135_bowtie_no_ori += 1
      else:
        count_135_bowtie_with_ori += 1

  count_GR45_soma0_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))

  no_ori_135_bowtie=count_135_bowtie_no_ori/(count_135_bowtie_no_ori+count_45_bowtie_no_ori)
  count_GR45_soma0_no_ori_135_bowtie_ratio.append(no_ori_135_bowtie)

  with_ori_135_bowtie=count_135_bowtie_with_ori/(count_135_bowtie_with_ori+count_45_bowtie_with_ori)
  count_GR45_soma0_with_ori_135_bowtie_ratio.append(with_ori_135_bowtie)

  deltafraction = with_ori_135_bowtie - no_ori_135_bowtie
  count_GR45_soma0_deltafraction.append(deltafraction)

    

##############################################################################################
## soma45 --> soma oridx = 6 --> \ #OVEREPRESENTED

for i in range(int(7*num_trials/10)):

  trial_ori = np.load(file_oh_6_GR45 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_6_GR45 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_6_GR45 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count_45_bowtie_no_ori = 0
  count_135_bowtie_no_ori = 0
  count_45_bowtie_with_ori = 0
  count_135_bowtie_with_ori = 0

  for ii in range(L):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
      if trial_ori[ii]==-1:
        count_45_bowtie_no_ori += 1
      else:
        count_45_bowtie_with_ori += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
      if trial_ori[ii]==-1:
        count_135_bowtie_no_ori += 1
      else:
        count_135_bowtie_with_ori += 1

  count_GR45_soma45_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))

  no_ori_135_bowtie=count_135_bowtie_no_ori/(count_135_bowtie_no_ori+count_45_bowtie_no_ori)
  count_GR45_soma45_no_ori_135_bowtie_ratio.append(no_ori_135_bowtie)

  with_ori_135_bowtie=count_135_bowtie_with_ori/(count_135_bowtie_with_ori+count_45_bowtie_with_ori)
  count_GR45_soma45_with_ori_135_bowtie_ratio.append(with_ori_135_bowtie)

  deltafraction = with_ori_135_bowtie - no_ori_135_bowtie
  count_GR45_soma45_deltafraction.append(deltafraction)
    


##############################################################################################
## soma90 --> soma oridx = 0 --> | vertical

for i in range(int(num_trials/10)):

  trial_ori = np.load(file_oh_0_GR45 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_0_GR45 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_0_GR45 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count_45_bowtie_no_ori = 0
  count_135_bowtie_no_ori = 0
  count_45_bowtie_with_ori = 0
  count_135_bowtie_with_ori = 0

  for ii in range(L):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
      if trial_ori[ii]==-1:
        count_45_bowtie_no_ori += 1
      else:
        count_45_bowtie_with_ori += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
      if trial_ori[ii]==-1:
        count_135_bowtie_no_ori += 1
      else:
        count_135_bowtie_with_ori += 1

  count_GR45_soma90_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))

  no_ori_135_bowtie=count_135_bowtie_no_ori/(count_135_bowtie_no_ori+count_45_bowtie_no_ori)
  count_GR45_soma90_no_ori_135_bowtie_ratio.append(no_ori_135_bowtie)

  with_ori_135_bowtie=count_135_bowtie_with_ori/(count_135_bowtie_with_ori+count_45_bowtie_with_ori)
  count_GR45_soma90_with_ori_135_bowtie_ratio.append(with_ori_135_bowtie)

  deltafraction = with_ori_135_bowtie - no_ori_135_bowtie
  count_GR45_soma90_deltafraction.append(deltafraction)
    


##############################################################################################
## soma135 --> soma oridx = 2 --> /

for i in range(int(num_trials/10)):

  trial_ori = np.load(file_oh_2_GR45 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_2_GR45 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_2_GR45 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count_45_bowtie_no_ori = 0
  count_135_bowtie_no_ori = 0
  count_45_bowtie_with_ori = 0
  count_135_bowtie_with_ori = 0

  for ii in range(L):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
      if trial_ori[ii]==-1:
        count_45_bowtie_no_ori += 1
      else:
        count_45_bowtie_with_ori += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
      if trial_ori[ii]==-1:
        count_135_bowtie_no_ori += 1
      else:
        count_135_bowtie_with_ori += 1

  count_GR45_soma135_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))

  no_ori_135_bowtie=count_135_bowtie_no_ori/(count_135_bowtie_no_ori+count_45_bowtie_no_ori)
  count_GR45_soma135_no_ori_135_bowtie_ratio.append(no_ori_135_bowtie)

  with_ori_135_bowtie=count_135_bowtie_with_ori/(count_135_bowtie_with_ori+count_45_bowtie_with_ori)
  count_GR45_soma135_with_ori_135_bowtie_ratio.append(with_ori_135_bowtie)

  deltafraction = with_ori_135_bowtie - no_ori_135_bowtie
  count_GR45_soma135_deltafraction.append(deltafraction)

    

#############################################
# Process and plot data set from simulation #
#                 GR135 Group               #
#############################################

file_oh_0_GR135 = file_oh_load + '/FB_GR135_soma_ori_0/trial'
file_oh_2_GR135 = file_oh_load + '/FB_GR135_soma_ori_2/trial'
file_oh_4_GR135 = file_oh_load + '/FB_GR135_soma_ori_4/trial'
file_oh_6_GR135 = file_oh_load + '/FB_GR135_soma_ori_6/trial'

## soma0 --> soma oridx = 4 --> -- horizontal
count_GR135_soma0_135_bowtie_ratio = []
count_GR135_soma0_no_ori_135_bowtie_ratio = []
count_GR135_soma0_with_ori_135_bowtie_ratio = []
count_GR135_soma0_deltafraction = []

## soma45 --> soma oridx = 6 --> \
count_GR135_soma45_135_bowtie_ratio = []
count_GR135_soma45_no_ori_135_bowtie_ratio = []
count_GR135_soma45_with_ori_135_bowtie_ratio = []
count_GR135_soma45_deltafraction = []

## soma90 --> soma oridx = 0 --> | vertical
count_GR135_soma90_135_bowtie_ratio = []
count_GR135_soma90_no_ori_135_bowtie_ratio = []
count_GR135_soma90_with_ori_135_bowtie_ratio = []
count_GR135_soma90_deltafraction = []

## soma135 --> soma oridx = 2 --> /
count_GR135_soma135_135_bowtie_ratio = []
count_GR135_soma135_no_ori_135_bowtie_ratio = []
count_GR135_soma135_with_ori_135_bowtie_ratio = []
count_GR135_soma135_deltafraction = []

##############################################################################################
## soma0 --> soma oridx = 4 --> -- horizontal

for i in range(int(num_trials/10)):

  trial_ori = np.load(file_oh_4_GR135 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_4_GR135 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_4_GR135 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count_45_bowtie_no_ori = 0
  count_135_bowtie_no_ori = 0
  count_45_bowtie_with_ori = 0
  count_135_bowtie_with_ori = 0

  for ii in range(L):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
      if trial_ori[ii]==-1:
        count_45_bowtie_no_ori += 1
      else:
        count_45_bowtie_with_ori += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
      if trial_ori[ii]==-1:
        count_135_bowtie_no_ori += 1
      else:
        count_135_bowtie_with_ori += 1

  count_GR135_soma0_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))

  no_ori_135_bowtie=count_135_bowtie_no_ori/(count_135_bowtie_no_ori+count_45_bowtie_no_ori)
  count_GR135_soma0_no_ori_135_bowtie_ratio.append(no_ori_135_bowtie)

  with_ori_135_bowtie=count_135_bowtie_with_ori/(count_135_bowtie_with_ori+count_45_bowtie_with_ori)
  count_GR135_soma0_with_ori_135_bowtie_ratio.append(with_ori_135_bowtie)

  deltafraction = with_ori_135_bowtie - no_ori_135_bowtie
  count_GR135_soma0_deltafraction.append(deltafraction)



##############################################################################################
## soma45 --> soma oridx = 6 --> \

for i in range(int(num_trials/10)):

  trial_ori = np.load(file_oh_6_GR135 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_6_GR135 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_6_GR135 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count_45_bowtie_no_ori = 0
  count_135_bowtie_no_ori = 0
  count_45_bowtie_with_ori = 0
  count_135_bowtie_with_ori = 0

  for ii in range(L):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
      if trial_ori[ii]==-1:
        count_45_bowtie_no_ori += 1
      else:
        count_45_bowtie_with_ori += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
      if trial_ori[ii]==-1:
        count_135_bowtie_no_ori += 1
      else:
        count_135_bowtie_with_ori += 1

  count_GR135_soma45_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))

  no_ori_135_bowtie=count_135_bowtie_no_ori/(count_135_bowtie_no_ori+count_45_bowtie_no_ori)
  count_GR135_soma45_no_ori_135_bowtie_ratio.append(no_ori_135_bowtie)

  with_ori_135_bowtie=count_135_bowtie_with_ori/(count_135_bowtie_with_ori+count_45_bowtie_with_ori)
  count_GR135_soma45_with_ori_135_bowtie_ratio.append(with_ori_135_bowtie)

  deltafraction = with_ori_135_bowtie - no_ori_135_bowtie
  count_GR135_soma45_deltafraction.append(deltafraction)
    


##############################################################################################
## soma90 --> soma oridx = 0 --> | vertical

for i in range(int(num_trials/10)):

  trial_ori = np.load(file_oh_0_GR135 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_0_GR135 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_0_GR135 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count_45_bowtie_no_ori = 0
  count_135_bowtie_no_ori = 0
  count_45_bowtie_with_ori = 0
  count_135_bowtie_with_ori = 0

  for ii in range(L):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
      if trial_ori[ii]==-1:
        count_45_bowtie_no_ori += 1
      else:
        count_45_bowtie_with_ori += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
      if trial_ori[ii]==-1:
        count_135_bowtie_no_ori += 1
      else:
        count_135_bowtie_with_ori += 1

  count_GR135_soma90_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))

  no_ori_135_bowtie=count_135_bowtie_no_ori/(count_135_bowtie_no_ori+count_45_bowtie_no_ori)
  count_GR135_soma90_no_ori_135_bowtie_ratio.append(no_ori_135_bowtie)

  with_ori_135_bowtie=count_135_bowtie_with_ori/(count_135_bowtie_with_ori+count_45_bowtie_with_ori)
  count_GR135_soma90_with_ori_135_bowtie_ratio.append(with_ori_135_bowtie)

  deltafraction = with_ori_135_bowtie - no_ori_135_bowtie
  count_GR135_soma90_deltafraction.append(deltafraction)
    


##############################################################################################
## soma135 --> soma oridx = 2 --> /  OVEREPRESENTED

for i in range(int(7*num_trials/10)):

  trial_ori = np.load(file_oh_2_GR135 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_2_GR135 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_2_GR135 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count_45_bowtie_no_ori = 0
  count_135_bowtie_no_ori = 0
  count_45_bowtie_with_ori = 0
  count_135_bowtie_with_ori = 0

  for ii in range(L):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
      if trial_ori[ii]==-1:
        count_45_bowtie_no_ori += 1
      else:
        count_45_bowtie_with_ori += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
      if trial_ori[ii]==-1:
        count_135_bowtie_no_ori += 1
      else:
        count_135_bowtie_with_ori += 1

  count_GR135_soma135_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))

  no_ori_135_bowtie=count_135_bowtie_no_ori/(count_135_bowtie_no_ori+count_45_bowtie_no_ori)
  count_GR135_soma135_no_ori_135_bowtie_ratio.append(no_ori_135_bowtie)

  with_ori_135_bowtie=count_135_bowtie_with_ori/(count_135_bowtie_with_ori+count_45_bowtie_with_ori)
  count_GR135_soma135_with_ori_135_bowtie_ratio.append(with_ori_135_bowtie)

  deltafraction = with_ori_135_bowtie - no_ori_135_bowtie
  count_GR135_soma135_deltafraction.append(deltafraction)


    
##########################################################
#          Generate box plot for all 3 groups            #
##########################################################

all_V1_CTRL_135_bowtie_ratio = count_CTRL_soma0_135_bowtie_ratio + count_CTRL_soma45_135_bowtie_ratio + count_CTRL_soma90_135_bowtie_ratio + count_CTRL_soma135_135_bowtie_ratio
all_V1_GR45_135_bowtie_ratio = count_GR45_soma0_135_bowtie_ratio + count_GR45_soma45_135_bowtie_ratio + count_GR45_soma90_135_bowtie_ratio + count_GR45_soma135_135_bowtie_ratio
all_V1_GR135_135_bowtie_ratio = count_GR135_soma0_135_bowtie_ratio + count_GR135_soma45_135_bowtie_ratio + count_GR135_soma90_135_bowtie_ratio + count_GR135_soma135_135_bowtie_ratio

all_V1_CTRL_135_no_ori_bowtie_ratio = count_CTRL_soma0_no_ori_135_bowtie_ratio + count_CTRL_soma45_no_ori_135_bowtie_ratio + count_CTRL_soma90_no_ori_135_bowtie_ratio + count_CTRL_soma135_no_ori_135_bowtie_ratio
all_V1_GR45_135_no_ori_bowtie_ratio = count_GR45_soma0_no_ori_135_bowtie_ratio + count_GR45_soma45_no_ori_135_bowtie_ratio + count_GR45_soma90_no_ori_135_bowtie_ratio + count_GR45_soma135_no_ori_135_bowtie_ratio
all_V1_GR135_135_no_ori_bowtie_ratio = count_GR135_soma0_no_ori_135_bowtie_ratio + count_GR135_soma45_no_ori_135_bowtie_ratio + count_GR135_soma90_no_ori_135_bowtie_ratio + count_GR135_soma135_no_ori_135_bowtie_ratio

all_V1_CTRL_135_with_ori_bowtie_ratio = count_CTRL_soma0_with_ori_135_bowtie_ratio + count_CTRL_soma45_with_ori_135_bowtie_ratio + count_CTRL_soma90_with_ori_135_bowtie_ratio + count_CTRL_soma135_with_ori_135_bowtie_ratio
all_V1_GR45_135_with_ori_bowtie_ratio = count_GR45_soma0_with_ori_135_bowtie_ratio + count_GR45_soma45_with_ori_135_bowtie_ratio + count_GR45_soma90_with_ori_135_bowtie_ratio + count_GR45_soma135_with_ori_135_bowtie_ratio
all_V1_GR135_135_with_ori_bowtie_ratio = count_GR135_soma0_with_ori_135_bowtie_ratio + count_GR135_soma45_with_ori_135_bowtie_ratio + count_GR135_soma90_with_ori_135_bowtie_ratio + count_GR135_soma135_with_ori_135_bowtie_ratio

all_V1_CTRL_deltafraction = count_CTRL_soma0_deltafraction + count_CTRL_soma45_deltafraction + count_CTRL_soma90_deltafraction + count_CTRL_soma135_deltafraction
all_V1_GR45_deltafraction = count_GR45_soma0_deltafraction + count_GR45_soma45_deltafraction + count_GR45_soma90_deltafraction + count_GR45_soma135_deltafraction
all_V1_GR135_deltafraction = count_GR135_soma0_deltafraction + count_GR135_soma45_deltafraction + count_GR135_soma90_deltafraction + count_GR135_soma135_deltafraction

all_FF_135_bowtie_ratio = all_V1_GR45_135_bowtie_ratio + all_V1_CTRL_135_bowtie_ratio + all_V1_GR135_135_bowtie_ratio
all_FF_135_no_ori_bowtie_ratio = all_V1_GR45_135_no_ori_bowtie_ratio + all_V1_CTRL_135_no_ori_bowtie_ratio + all_V1_GR135_135_no_ori_bowtie_ratio
all_FF_135_with_ori_bowtie_ratio = all_V1_GR45_135_with_ori_bowtie_ratio + all_V1_CTRL_135_with_ori_bowtie_ratio + all_V1_GR135_135_with_ori_bowtie_ratio
all_FF_deltafraction = all_V1_GR45_deltafraction + all_V1_CTRL_deltafraction + all_V1_GR135_deltafraction


groups = ['GR45'] * num_trials + ['CTRL'] * num_trials + ['GR135'] * num_trials
all_FF_135_bowtie_ratio_dataset = {'Group': groups, 
                                    'Boutons in 135deg bowtie ratio': all_FF_135_bowtie_ratio,
                                    'No orientation Boutons in 135deg bowtie ratio': all_FF_135_no_ori_bowtie_ratio, 
                                    'With orientation Boutons in 135deg bowtie ratio': all_FF_135_with_ori_bowtie_ratio,
                                    'DeltaFraction': all_FF_deltafraction}


df_all_FF_135_bowtie_ratio = pd.DataFrame(all_FF_135_bowtie_ratio_dataset)

t_stat_GR45_GR135, p_val_GR45_GR135 = stats.ttest_ind(all_V1_GR45_135_bowtie_ratio, all_V1_GR135_135_bowtie_ratio, equal_var=False)
t_stat_CTRL_GR135, p_val_CTRL_GR135 = stats.ttest_ind(all_V1_CTRL_135_bowtie_ratio, all_V1_GR135_135_bowtie_ratio, equal_var=False)
t_stat_CTRL_GR45, p_val_CTRL_GR45 = stats.ttest_ind(all_V1_CTRL_135_bowtie_ratio, all_V1_GR45_135_bowtie_ratio, equal_var=False)

t_stat_GR45_GR135_no_ori, p_val_GR45_GR135_no_ori = stats.ttest_ind(all_V1_GR45_135_no_ori_bowtie_ratio, all_V1_GR135_135_no_ori_bowtie_ratio, equal_var=False)
t_stat_CTRL_GR135_no_ori, p_val_CTRL_GR135_no_ori = stats.ttest_ind(all_V1_CTRL_135_no_ori_bowtie_ratio, all_V1_GR135_135_no_ori_bowtie_ratio, equal_var=False)
t_stat_CTRL_GR45_no_ori, p_val_CTRL_GR45_no_ori = stats.ttest_ind(all_V1_CTRL_135_no_ori_bowtie_ratio, all_V1_GR45_135_no_ori_bowtie_ratio, equal_var=False)

t_stat_GR45_GR135_with_ori, p_val_GR45_GR135_with_ori = stats.ttest_ind(all_V1_GR45_135_with_ori_bowtie_ratio, all_V1_GR135_135_with_ori_bowtie_ratio, equal_var=False)
t_stat_CTRL_GR135_with_ori, p_val_CTRL_GR135_with_ori = stats.ttest_ind(all_V1_CTRL_135_with_ori_bowtie_ratio, all_V1_GR135_135_with_ori_bowtie_ratio, equal_var=False)
t_stat_CTRL_GR45_with_ori, p_val_CTRL_GR45_with_ori = stats.ttest_ind(all_V1_CTRL_135_with_ori_bowtie_ratio, all_V1_GR45_135_with_ori_bowtie_ratio, equal_var=False)

t_stat_GR45_GR135_deltafraction, p_val_GR45_GR135_deltafraction = stats.ttest_ind(all_V1_GR45_deltafraction, all_V1_GR135_deltafraction, equal_var=False)
t_stat_CTRL_GR135_deltafraction, p_val_CTRL_GR135_deltafraction = stats.ttest_ind(all_V1_CTRL_deltafraction, all_V1_GR135_deltafraction, equal_var=False)
t_stat_CTRL_GR45_deltafraction, p_val_CTRL_GR45_deltafraction = stats.ttest_ind(all_V1_CTRL_deltafraction, all_V1_GR45_deltafraction, equal_var=False)

custom_palette = ["darkviolet", "sienna", "deeppink"]


###################PLOT FOR ALL BOUTONS###################
plt.figure()
sns.stripplot(data=df_all_FF_135_bowtie_ratio, x="Group", y="Boutons in 135deg bowtie ratio", alpha=.1, legend=False, hue = 'Group', palette = custom_palette, jitter=False)
g = sns.pointplot(data=df_all_FF_135_bowtie_ratio, x="Group", y="Boutons in 135deg bowtie ratio", errorbar=('ci', 95), marker="_", markersize=25, markeredgewidth=20, hue = 'Group', palette = custom_palette)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

annotations = [("GR45", "CTRL", p_val_CTRL_GR45), ("GR135", "CTRL", p_val_CTRL_GR135)]
starbars.draw_annotation(annotations)
annotations = [("GR45", "GR135", p_val_GR45_GR135)]
starbars.draw_annotation(annotations)

ax = g.axes
ax.set_ylabel('Fraction of boutons in the 135º bowtie')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=0, length=0)
sns.boxplot(data=df_all_FF_135_bowtie_ratio, x="Group", y="Boutons in 135deg bowtie ratio", legend=False, hue = 'Group', palette = custom_palette, width=0.4, linewidth=1.5)

sns.despine(right=True, top=True)
plt.ylim([0.2,0.8])
plt.tight_layout()

sp = 0.5
ax.axhline(y=sp, color='gray', linestyle='--')

plt.savefig(file_oh_save + 'V1_FBRF_LM_bouton_ratio_in_135_deg_bowtie.png', format='png', dpi=200)

###################PLOT FOR ALL BOUTONS###################

###################PLOT FOR ALL NO ORI BOUTONS###################
plt.figure()
sns.stripplot(data=df_all_FF_135_bowtie_ratio, x="Group", y="No orientation Boutons in 135deg bowtie ratio", alpha=.1, legend=False, hue = 'Group', palette = custom_palette, jitter=False)
g = sns.pointplot(data=df_all_FF_135_bowtie_ratio, x="Group", y="No orientation Boutons in 135deg bowtie ratio", errorbar=('ci', 95), marker="_", markersize=25, markeredgewidth=20, hue = 'Group', palette = custom_palette)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

annotations = [("GR45", "CTRL", p_val_CTRL_GR45_no_ori), ("GR135", "CTRL", p_val_CTRL_GR135_no_ori)]
starbars.draw_annotation(annotations)
annotations = [("GR45", "GR135", p_val_GR45_GR135_no_ori)]
starbars.draw_annotation(annotations)

ax = g.axes
ax.set_ylabel('Fraction of non-orientation responsive boutons in the 135º bowtie')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=0, length=0)
sns.boxplot(data=df_all_FF_135_bowtie_ratio, x="Group", y="No orientation Boutons in 135deg bowtie ratio", legend=False, hue = 'Group', palette = custom_palette, width=0.4, linewidth=1.5)

sns.despine(right=True, top=True)
plt.ylim([0.2,0.8])
plt.tight_layout()

sp = 0.5
ax.axhline(y=sp, color='gray', linestyle='--')

plt.savefig(file_oh_save + 'V1_FBRF_LM_bouton_ratio_in_135_deg_bowtie_no_res.png', format='png', dpi=200)

###################PLOT FOR ALL NO ORI BOUTONS###################

###################PLOT FOR ALL ORI BOUTONS###################
plt.figure()
sns.stripplot(data=df_all_FF_135_bowtie_ratio, x="Group", y="With orientation Boutons in 135deg bowtie ratio", alpha=.1, legend=False, hue = 'Group', palette = custom_palette, jitter=False)
g = sns.pointplot(data=df_all_FF_135_bowtie_ratio, x="Group", y="With orientation Boutons in 135deg bowtie ratio", errorbar=('ci', 95), marker="_", markersize=25, markeredgewidth=20, hue = 'Group', palette = custom_palette)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

annotations = [("GR45", "CTRL", p_val_CTRL_GR45_with_ori), ("GR135", "CTRL", p_val_CTRL_GR135_with_ori)]
starbars.draw_annotation(annotations)
annotations = [("GR45", "GR135", p_val_GR45_GR135_with_ori)]
starbars.draw_annotation(annotations)

ax = g.axes
ax.set_ylabel('Fraction of orientation responsive boutons in the 135º bowtie')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=0, length=0)
sns.boxplot(data=df_all_FF_135_bowtie_ratio, x="Group", y="With orientation Boutons in 135deg bowtie ratio", legend=False, hue = 'Group', palette = custom_palette, width=0.4, linewidth=1.5)

sns.despine(right=True, top=True)
plt.ylim([0.2,0.8])
plt.tight_layout()

sp = 0.5
ax.axhline(y=sp, color='gray', linestyle='--')

plt.savefig(file_oh_save + 'V1_FBRF_LM_bouton_ratio_in_135_deg_bowtie_res.png', format='png', dpi=200)

###################PLOT FOR ALL ORI BOUTONS###################

###################PLOT FOR DELTAFRACTIONS###################
plt.figure()
sns.stripplot(data=df_all_FF_135_bowtie_ratio, x="Group", y='DeltaFraction', alpha=.1, legend=False, hue = 'Group', palette = custom_palette, jitter=False)
g = sns.pointplot(data=df_all_FF_135_bowtie_ratio, x="Group", y='DeltaFraction', errorbar=('ci', 95), marker="_", markersize=25, markeredgewidth=20, hue = 'Group', palette = custom_palette)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

annotations = [("GR45", "CTRL", p_val_CTRL_GR45_deltafraction), ("GR135", "CTRL", p_val_CTRL_GR135_deltafraction)]
starbars.draw_annotation(annotations)
annotations = [("GR45", "GR135", p_val_GR45_GR135_deltafraction)]
starbars.draw_annotation(annotations)

ax = g.axes
ax.set_ylabel('ΔFraction')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=0, length=0)
sns.boxplot(data=df_all_FF_135_bowtie_ratio, x="Group", y='DeltaFraction', legend=False, hue = 'Group', palette = custom_palette, width=0.4, linewidth=1.5)

sns.despine(right=True, top=True)
plt.ylim([-0.5,0.5])
plt.tight_layout()

sp = 0.0
ax.axhline(y=sp, color='gray', linestyle='--')

plt.savefig(file_oh_save + 'V1_FBRF_LM_deltafractions.png', format='png', dpi=200)

###################PLOT FOR DELTAFRACTIONS###################

# ---- tidy data: two columns -> one long frame ----
long = pd.melt(
    df_all_FF_135_bowtie_ratio,
    id_vars=["Group"],
    value_vars=[
        "With orientation Boutons in 135deg bowtie ratio",
        "No orientation Boutons in 135deg bowtie ratio",
    ],
    var_name="Type",
    value_name="Ratio",
).replace({
    "With orientation Boutons in 135deg bowtie ratio": "Responsive",
    "No orientation Boutons in 135deg bowtie ratio": "Non-responsive",
})

group_order = ["GR45", "CTRL", "GR135"]       # pair 1, 2, 3
type_order  = ["Responsive", "Non-responsive"] # left, right within each pair


# ---- plot (matplotlib boxplot per pair) ----
fig, ax = plt.subplots(figsize=(8,5))

pair_centers = np.arange(1, len(group_order)+1)      # 1,2,3
half_spacing  = 0.20                                  # left/right offset inside a pair
x_positions   = {
    "Responsive":     pair_centers - half_spacing,
    "Non-responsive": pair_centers + half_spacing,
}

# draw 3 pairs: one per group, 2 boxes per pair, both in the group color
group_palette = make_group_palette(custom_palette, group_order)
for i, g in enumerate(group_order, start=1):
    gdata = long[long["Group"] == g]
    color = group_palette[g]

    # left box: Responsive
    left_vals = gdata[gdata["Type"]=="Responsive"]["Ratio"].dropna().values
    if len(left_vals):
        bp = ax.boxplot(
            [left_vals], positions=[x_positions["Responsive"][i-1]],
            widths=0.35, patch_artist=True, manage_ticks=False,
            whiskerprops=dict(color="black"), capprops=dict(color="black"),
            medianprops=dict(color="black"), boxprops=dict(facecolor=color, edgecolor="black")
        )

    # right box: Non-responsive
    right_vals = gdata[gdata["Type"]=="Non-responsive"]["Ratio"].dropna().values
    if len(right_vals):
        bp = ax.boxplot(
            [right_vals], positions=[x_positions["Non-responsive"][i-1]],
            widths=0.35, patch_artist=True, manage_ticks=False,
            whiskerprops=dict(color="black"), capprops=dict(color="black"),
            medianprops=dict(color="black"), boxprops=dict(facecolor=color, edgecolor="black"),
        )

    # connect means for this pair, in the same group color
    if len(left_vals) and len(right_vals):
        yl = np.median(left_vals)
        yr = np.median(right_vals)
        xl = x_positions["Responsive"][i-1]
        xr = x_positions["Non-responsive"][i-1]
        ax.plot([xl, xr], [yl, yr], color="k", linewidth=2, zorder=5)
        ax.scatter([xl, xr], [yl, yr], color="k", s=35, zorder=6)

# cosmetics
ax.set_xlim(0.5, len(group_order)+0.5)
ax.set_ylim(0.2, 0.8)
ax.axhline(0.5, color="gray", linestyle="--", zorder=0)

# x-axis should say the same label under each pair:
ax.set_xticks(pair_centers)
ax.set_xticklabels(["responsive – non-responsive"] * len(group_order))
ax.set_xlabel("")
ax.set_ylabel("Fraction of boutons in the 135º bowtie")

# legend shows which color = which Group
handles = [plt.Line2D([0],[0], color=group_palette[g], lw=6, label=g) for g in group_order]
ax.legend(handles=handles, title="Group", frameon=False, loc="lower left")

sns.despine(right=True, top=True)
plt.tight_layout()

plt.savefig(file_oh_save + 'V1_FBRF_LM_bouton_ratio_in_135_deg_bowtie_res_no_res.png', format='png', dpi=200)
