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
-"V1_FBRF_LM_bouton_ratio_in_135_deg_bowtie.png" -> Boxplots for the ratio of boutons in
the 135 degree bowtie of all three groups, statistically compared.

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
count_CTRL_soma0_45_bowtie = []
count_CTRL_soma0_135_bowtie = []
count45bt_CTRL_soma0_45_bowtie = []
count45bt_CTRL_soma0_135_bowtie = []
count135bt_CTRL_soma0_45_bowtie = []
count135bt_CTRL_soma0_135_bowtie = []
count_CTRL_soma0_135_bowtie_ratio = []

## soma45 --> soma oridx = 6 --> \
count_CTRL_soma45_45_bowtie = []
count_CTRL_soma45_135_bowtie = []
count45bt_CTRL_soma45_45_bowtie = []
count45bt_CTRL_soma45_135_bowtie = []
count135bt_CTRL_soma45_45_bowtie = []
count135bt_CTRL_soma45_135_bowtie = []
count_CTRL_soma45_135_bowtie_ratio = []

## soma90 --> soma oridx = 0 --> | vertical
count_CTRL_soma90_45_bowtie = []
count_CTRL_soma90_135_bowtie = []
count45bt_CTRL_soma90_45_bowtie = []
count45bt_CTRL_soma90_135_bowtie = []
count135bt_CTRL_soma90_45_bowtie = []
count135bt_CTRL_soma90_135_bowtie = []
count_CTRL_soma90_135_bowtie_ratio = []

## soma135 --> soma oridx = 2 --> /
count_CTRL_soma135_45_bowtie = []
count_CTRL_soma135_135_bowtie = []
count45bt_CTRL_soma135_45_bowtie = []
count45bt_CTRL_soma135_135_bowtie = []
count135bt_CTRL_soma135_45_bowtie = []
count135bt_CTRL_soma135_135_bowtie = []
count_CTRL_soma135_135_bowtie_ratio = []

isoori_avg = 0
colinear_ori = [5,6,7]
w_th = 0.2


##############################################################################################
## soma: 0 deg --> soma oridx = 4 --> -- horizontal

for i in range(int(num_trials/4)):

  trial_ori = np.load(file_oh_CTRL_4 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_CTRL_4 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_CTRL_4 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count45bt_45_bowtie = 0
  count45bt_135_bowtie = 0
  count135bt_45_bowtie = 0
  count135bt_135_bowtie = 0

  for ii in range(100):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_135_bowtie += 1
  count_CTRL_soma0_45_bowtie.append(count_45_bowtie)
  count_CTRL_soma0_135_bowtie.append(count_135_bowtie)
  count45bt_CTRL_soma0_45_bowtie.append(count45bt_45_bowtie)
  count45bt_CTRL_soma0_135_bowtie.append(count45bt_135_bowtie)
  count135bt_CTRL_soma0_45_bowtie.append(count135bt_45_bowtie)
  count135bt_CTRL_soma0_135_bowtie.append(count135bt_135_bowtie)
  count_CTRL_soma0_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))


##############################################################################################
## soma: 45 deg --> soma oridx = 6 --> \

for i in range(int(num_trials/4)):

  trial_ori = np.load(file_oh_CTRL_6 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_CTRL_6 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_CTRL_6 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count45bt_45_bowtie = 0
  count45bt_135_bowtie = 0
  count135bt_45_bowtie = 0
  count135bt_135_bowtie = 0

  for ii in range(100):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_135_bowtie += 1
  count_CTRL_soma45_45_bowtie.append(count_45_bowtie)
  count_CTRL_soma45_135_bowtie.append(count_135_bowtie)
  count45bt_CTRL_soma45_45_bowtie.append(count45bt_45_bowtie)
  count45bt_CTRL_soma45_135_bowtie.append(count45bt_135_bowtie)
  count135bt_CTRL_soma45_45_bowtie.append(count135bt_45_bowtie)
  count135bt_CTRL_soma45_135_bowtie.append(count135bt_135_bowtie)
  count_CTRL_soma45_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))

##############################################################################################
## soma: 90 deg --> soma oridx = 0 --> | vertical

for i in range(int(num_trials/4)):

  trial_ori = np.load(file_oh_CTRL_0 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_CTRL_0 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_CTRL_0 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count45bt_45_bowtie = 0
  count45bt_135_bowtie = 0
  count135bt_45_bowtie = 0
  count135bt_135_bowtie = 0

  for ii in range(100):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_135_bowtie += 1
  count_CTRL_soma90_45_bowtie.append(count_45_bowtie)
  count_CTRL_soma90_135_bowtie.append(count_135_bowtie)
  count45bt_CTRL_soma90_45_bowtie.append(count45bt_45_bowtie)
  count45bt_CTRL_soma90_135_bowtie.append(count45bt_135_bowtie)
  count135bt_CTRL_soma90_45_bowtie.append(count135bt_45_bowtie)
  count135bt_CTRL_soma90_135_bowtie.append(count135bt_135_bowtie)
  count_CTRL_soma90_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))

##############################################################################################
## soma: 135 deg --> soma oridx = 2 --> /

for i in range(int(num_trials/4)):

  trial_ori = np.load(file_oh_CTRL_2 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_CTRL_2 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_CTRL_2 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count45bt_45_bowtie = 0
  count45bt_135_bowtie = 0
  count135bt_45_bowtie = 0
  count135bt_135_bowtie = 0

  for ii in range(100):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_135_bowtie += 1
  count_CTRL_soma135_45_bowtie.append(count_45_bowtie)
  count_CTRL_soma135_135_bowtie.append(count_135_bowtie)
  count45bt_CTRL_soma135_45_bowtie.append(count45bt_45_bowtie)
  count45bt_CTRL_soma135_135_bowtie.append(count45bt_135_bowtie)
  count135bt_CTRL_soma135_45_bowtie.append(count135bt_45_bowtie)
  count135bt_CTRL_soma135_135_bowtie.append(count135bt_135_bowtie)
  count_CTRL_soma135_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))

#############################################
# Process and plot data set from simulation #
#                 GR45 Group                #
#############################################
  

file_oh_0_GR45 = file_oh_load + '/FB_GR45_soma_ori_0/trial'
file_oh_2_GR45 = file_oh_load + '/FB_GR45_soma_ori_2/trial'
file_oh_4_GR45 = file_oh_load + '/FB_GR45_soma_ori_4/trial'
file_oh_6_GR45 = file_oh_load + '/FB_GR45_soma_ori_6/trial'

## soma0 --> soma oridx = 4 --> -- horizontal
count_GR45_soma0_45_bowtie = []
count_GR45_soma0_135_bowtie = []
count45bt_GR45_soma0_45_bowtie = []
count45bt_GR45_soma0_135_bowtie = []
count135bt_GR45_soma0_45_bowtie = []
count135bt_GR45_soma0_135_bowtie = []
count_GR45_soma0_135_bowtie_ratio = []
## soma45 --> soma oridx = 6 --> \
count_GR45_soma45_45_bowtie = []
count_GR45_soma45_135_bowtie = []
count45bt_GR45_soma45_45_bowtie = []
count45bt_GR45_soma45_135_bowtie = []
count135bt_GR45_soma45_45_bowtie = []
count135bt_GR45_soma45_135_bowtie = []
count_GR45_soma45_135_bowtie_ratio = []
## soma90 --> soma oridx = 0 --> | vertical
count_GR45_soma90_45_bowtie = []
count_GR45_soma90_135_bowtie = []
count45bt_GR45_soma90_45_bowtie = []
count45bt_GR45_soma90_135_bowtie = []
count135bt_GR45_soma90_45_bowtie = []
count135bt_GR45_soma90_135_bowtie = []
count_GR45_soma90_135_bowtie_ratio = []
## soma135 --> soma oridx = 2 --> /
count_GR45_soma135_45_bowtie = []
count_GR45_soma135_135_bowtie = []
count45bt_GR45_soma135_45_bowtie = []
count45bt_GR45_soma135_135_bowtie = []
count135bt_GR45_soma135_45_bowtie = []
count135bt_GR45_soma135_135_bowtie = []
count_GR45_soma135_135_bowtie_ratio = []

isoori_avg = 0
colinear_ori = [5,6,7]
w_th = 0.2


##############################################################################################
## soma0 --> soma oridx = 4 --> -- horizontal

for i in range(int(num_trials/10)):

  trial_ori = np.load(file_oh_4_GR45 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_4_GR45 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_4_GR45 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count45bt_45_bowtie = 0
  count45bt_135_bowtie = 0
  count135bt_45_bowtie = 0
  count135bt_135_bowtie = 0

  for ii in range(100):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_135_bowtie += 1
  count_GR45_soma0_45_bowtie.append(count_45_bowtie)
  count_GR45_soma0_135_bowtie.append(count_135_bowtie)
  count45bt_GR45_soma0_45_bowtie.append(count45bt_45_bowtie)
  count45bt_GR45_soma0_135_bowtie.append(count45bt_135_bowtie)
  count135bt_GR45_soma0_45_bowtie.append(count135bt_45_bowtie)
  count135bt_GR45_soma0_135_bowtie.append(count135bt_135_bowtie)
  count_GR45_soma0_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))



##############################################################################################
## soma45 --> soma oridx = 6 --> \ #OVEREPRESENTED

for i in range(int(7*num_trials/10)):

  trial_ori = np.load(file_oh_6_GR45 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_6_GR45 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_6_GR45 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count45bt_45_bowtie = 0
  count45bt_135_bowtie = 0
  count135bt_45_bowtie = 0
  count135bt_135_bowtie = 0

  for ii in range(100):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_135_bowtie += 1
  count_GR45_soma45_45_bowtie.append(count_45_bowtie)
  count_GR45_soma45_135_bowtie.append(count_135_bowtie)
  count45bt_GR45_soma45_45_bowtie.append(count45bt_45_bowtie)
  count45bt_GR45_soma45_135_bowtie.append(count45bt_135_bowtie)
  count135bt_GR45_soma45_45_bowtie.append(count135bt_45_bowtie)
  count135bt_GR45_soma45_135_bowtie.append(count135bt_135_bowtie)
  count_GR45_soma45_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))



##############################################################################################
## soma90 --> soma oridx = 0 --> | vertical

for i in range(int(num_trials/10)):

  trial_ori = np.load(file_oh_0_GR45 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_0_GR45 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_0_GR45 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count45bt_45_bowtie = 0
  count45bt_135_bowtie = 0
  count135bt_45_bowtie = 0
  count135bt_135_bowtie = 0

  for ii in range(100):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_135_bowtie += 1
  count_GR45_soma90_45_bowtie.append(count_45_bowtie)
  count_GR45_soma90_135_bowtie.append(count_135_bowtie)
  count45bt_GR45_soma90_45_bowtie.append(count45bt_45_bowtie)
  count45bt_GR45_soma90_135_bowtie.append(count45bt_135_bowtie)
  count135bt_GR45_soma90_45_bowtie.append(count135bt_45_bowtie)
  count135bt_GR45_soma90_135_bowtie.append(count135bt_135_bowtie)
  count_GR45_soma90_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))



##############################################################################################
## soma135 --> soma oridx = 2 --> /

for i in range(int(num_trials/10)):

  trial_ori = np.load(file_oh_2_GR45 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_2_GR45 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_2_GR45 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count45bt_45_bowtie = 0
  count45bt_135_bowtie = 0
  count135bt_45_bowtie = 0
  count135bt_135_bowtie = 0

  for ii in range(100):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_135_bowtie += 1
  count_GR45_soma135_45_bowtie.append(count_45_bowtie)
  count_GR45_soma135_135_bowtie.append(count_135_bowtie)
  count45bt_GR45_soma135_45_bowtie.append(count45bt_45_bowtie)
  count45bt_GR45_soma135_135_bowtie.append(count45bt_135_bowtie)
  count135bt_GR45_soma135_45_bowtie.append(count135bt_45_bowtie)
  count135bt_GR45_soma135_135_bowtie.append(count135bt_135_bowtie)
  count_GR45_soma135_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))

#############################################
# Process and plot data set from simulation #
#                 GR135 Group               #
#############################################

file_oh_0_GR135 = file_oh_load + '/FB_GR135_soma_ori_0/trial'
file_oh_2_GR135 = file_oh_load + '/FB_GR135_soma_ori_2/trial'
file_oh_4_GR135 = file_oh_load + '/FB_GR135_soma_ori_4/trial'
file_oh_6_GR135 = file_oh_load + '/FB_GR135_soma_ori_6/trial'

## soma0 --> soma oridx = 4 --> -- horizontal
count_GR135_soma0_45_bowtie = []
count_GR135_soma0_135_bowtie = []
count45bt_GR135_soma0_45_bowtie = []
count45bt_GR135_soma0_135_bowtie = []
count135bt_GR135_soma0_45_bowtie = []
count135bt_GR135_soma0_135_bowtie = []
count_GR135_soma0_135_bowtie_ratio = []
## soma45 --> soma oridx = 6 --> \
count_GR135_soma45_45_bowtie = []
count_GR135_soma45_135_bowtie = []
count45bt_GR135_soma45_45_bowtie = []
count45bt_GR135_soma45_135_bowtie = []
count135bt_GR135_soma45_45_bowtie = []
count135bt_GR135_soma45_135_bowtie = []
count_GR135_soma45_135_bowtie_ratio = []
## soma90 --> soma oridx = 0 --> | vertical
count_GR135_soma90_45_bowtie = []
count_GR135_soma90_135_bowtie = []
count45bt_GR135_soma90_45_bowtie = []
count45bt_GR135_soma90_135_bowtie = []
count135bt_GR135_soma90_45_bowtie = []
count135bt_GR135_soma90_135_bowtie = []
count_GR135_soma90_135_bowtie_ratio = []
## soma135 --> soma oridx = 2 --> /
count_GR135_soma135_45_bowtie = []
count_GR135_soma135_135_bowtie = []
count45bt_GR135_soma135_45_bowtie = []
count45bt_GR135_soma135_135_bowtie = []
count135bt_GR135_soma135_45_bowtie = []
count135bt_GR135_soma135_135_bowtie = []
count_GR135_soma135_135_bowtie_ratio = []

isoori_avg = 0
colinear_ori = [5,6,7]
w_th = 0.2


##############################################################################################
## soma0 --> soma oridx = 4 --> -- horizontal

for i in range(int(num_trials/10)):

  trial_ori = np.load(file_oh_4_GR135 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_4_GR135 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_4_GR135 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count45bt_45_bowtie = 0
  count45bt_135_bowtie = 0
  count135bt_45_bowtie = 0
  count135bt_135_bowtie = 0

  for ii in range(100):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_135_bowtie += 1
  count_GR135_soma0_45_bowtie.append(count_45_bowtie)
  count_GR135_soma0_135_bowtie.append(count_135_bowtie)
  count45bt_GR135_soma0_45_bowtie.append(count45bt_45_bowtie)
  count45bt_GR135_soma0_135_bowtie.append(count45bt_135_bowtie)
  count135bt_GR135_soma0_45_bowtie.append(count135bt_45_bowtie)
  count135bt_GR135_soma0_135_bowtie.append(count135bt_135_bowtie)
  count_GR135_soma0_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))



##############################################################################################
## soma45 --> soma oridx = 6 --> \

for i in range(int(num_trials/10)):

  trial_ori = np.load(file_oh_6_GR135 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_6_GR135 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_6_GR135 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count45bt_45_bowtie = 0
  count45bt_135_bowtie = 0
  count135bt_45_bowtie = 0
  count135bt_135_bowtie = 0

  for ii in range(100):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_135_bowtie += 1
  count_GR135_soma45_45_bowtie.append(count_45_bowtie)
  count_GR135_soma45_135_bowtie.append(count_135_bowtie)
  count45bt_GR135_soma45_45_bowtie.append(count45bt_45_bowtie)
  count45bt_GR135_soma45_135_bowtie.append(count45bt_135_bowtie)
  count135bt_GR135_soma45_45_bowtie.append(count135bt_45_bowtie)
  count135bt_GR135_soma45_135_bowtie.append(count135bt_135_bowtie)
  count_GR135_soma45_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))



##############################################################################################
## soma90 --> soma oridx = 0 --> | vertical

for i in range(int(num_trials/10)):

  trial_ori = np.load(file_oh_0_GR135 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_0_GR135 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_0_GR135 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count45bt_45_bowtie = 0
  count45bt_135_bowtie = 0
  count135bt_45_bowtie = 0
  count135bt_135_bowtie = 0

  for ii in range(100):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_135_bowtie += 1
  count_GR135_soma90_45_bowtie.append(count_45_bowtie)
  count_GR135_soma90_135_bowtie.append(count_135_bowtie)
  count45bt_GR135_soma90_45_bowtie.append(count45bt_45_bowtie)
  count45bt_GR135_soma90_135_bowtie.append(count45bt_135_bowtie)
  count135bt_GR135_soma90_45_bowtie.append(count135bt_45_bowtie)
  count135bt_GR135_soma90_135_bowtie.append(count135bt_135_bowtie)
  count_GR135_soma90_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))



##############################################################################################
## soma135 --> soma oridx = 2 --> /  OVEREPRESENTED

for i in range(int(7*num_trials/10)):

  trial_ori = np.load(file_oh_2_GR135 + str(i) + '_ori.npy')
  trial_ori = trial_ori.astype(int)
  trial_coord = np.load(file_oh_2_GR135 + str(i) + '_coord.npy')
  trial_weight = np.load(file_oh_2_GR135 + str(i) + '_weight.npy')

  count_45_bowtie = 0
  count_135_bowtie = 0
  count45bt_45_bowtie = 0
  count45bt_135_bowtie = 0
  count135bt_45_bowtie = 0
  count135bt_135_bowtie = 0

  for ii in range(100):
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and trial_weight[ii] > w_th:
      count_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [5,6,7])):
      count45bt_135_bowtie += 1
    if ((trial_coord[ii,0] < 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] > 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_45_bowtie += 1
    if ((trial_coord[ii,0] > 0.5 and trial_coord[ii,1] < 0.5) or (trial_coord[ii,0] < 0.5 and trial_coord[ii,1] > 0.5)) and (trial_weight[ii] > w_th and (trial_ori[ii] in [1,2,3])):
      count135bt_135_bowtie += 1
  count_GR135_soma135_45_bowtie.append(count_45_bowtie)
  count_GR135_soma135_135_bowtie.append(count_135_bowtie)
  count45bt_GR135_soma135_45_bowtie.append(count45bt_45_bowtie)
  count45bt_GR135_soma135_135_bowtie.append(count45bt_135_bowtie)
  count135bt_GR135_soma135_45_bowtie.append(count135bt_45_bowtie)
  count135bt_GR135_soma135_135_bowtie.append(count135bt_135_bowtie)
  count_GR135_soma135_135_bowtie_ratio.append(count_135_bowtie/(count_135_bowtie+count_45_bowtie))


##########################################################
#          Generate box plot for all 3 groups            #
##########################################################

all_V1_CTRL_135_bowtie_ratio = count_CTRL_soma0_135_bowtie_ratio + count_CTRL_soma45_135_bowtie_ratio + count_CTRL_soma90_135_bowtie_ratio + count_CTRL_soma135_135_bowtie_ratio
all_V1_GR45_135_bowtie_ratio = count_GR45_soma0_135_bowtie_ratio + count_GR45_soma45_135_bowtie_ratio + count_GR45_soma90_135_bowtie_ratio + count_GR45_soma135_135_bowtie_ratio
all_V1_GR135_135_bowtie_ratio = count_GR135_soma0_135_bowtie_ratio + count_GR135_soma45_135_bowtie_ratio + count_GR135_soma90_135_bowtie_ratio + count_GR135_soma135_135_bowtie_ratio

all_FF_135_bowtie_ratio = all_V1_GR45_135_bowtie_ratio + all_V1_CTRL_135_bowtie_ratio + all_V1_GR135_135_bowtie_ratio


groups = ['GR45'] * num_trials + ['CTRL'] * num_trials + ['GR135'] * num_trials
all_FF_135_bowtie_ratio_dataset = {'Group': groups, 'Boutons in 135deg bowtie ratio': all_FF_135_bowtie_ratio}
df_all_FF_135_bowtie_ratio = pd.DataFrame(all_FF_135_bowtie_ratio_dataset)

t_stat_GR45_GR135, p_val_GR45_GR135 = stats.ttest_ind(all_V1_GR45_135_bowtie_ratio, all_V1_GR135_135_bowtie_ratio, equal_var=False)
t_stat_CTRL_GR135, p_val_CTRL_GR135 = stats.ttest_ind(all_V1_CTRL_135_bowtie_ratio, all_V1_GR135_135_bowtie_ratio, equal_var=False)
t_stat_CTRL_GR45, p_val_CTRL_GR45 = stats.ttest_ind(all_V1_CTRL_135_bowtie_ratio, all_V1_GR45_135_bowtie_ratio, equal_var=False)

custom_palette = ["darkviolet", "sienna", "deeppink"]

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
ax.set_ylabel('Ratio of Boutons in 135deg bowtie')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=0, length=0)
sns.boxplot(data=df_all_FF_135_bowtie_ratio, x="Group", y="Boutons in 135deg bowtie ratio", legend=False, hue = 'Group', palette = custom_palette, width=0.4, linewidth=1.5)

sns.despine(right=True, top=True)
plt.ylim([0.2,0.8])
plt.tight_layout()

sp = 0.5
ax.axhline(y=sp, color='gray', linestyle='--')

plt.savefig(file_oh_save + 'V1_FBRF_LM_bouton_ratio_in_135_deg_bowtie.png', format='png', dpi=200)