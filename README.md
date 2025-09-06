# Visual experience exerts an instructive role on cortical feedback inputs to primary visual cortex.
by Rodrigo F. Dias, Radhika Rajan, Xinyun Zhang, Margarida Baeta, Nikos Malakasis, Julijana Gjorgjieva and Leopoldo Petreanu

## Overview
Code writen by Xinyun Zhang (xy.zhang@tum.de) and Nikos Malakasis (nikos.malakasis@tum.de).

This repository contains the computational models developed for the study. The two models simulate how visual experience shapes: 

1. feedforward receptive fields and their orientation selectivity, of neurons in the latero-medial visual cortex (LM), through inputs from the primary visual cortex (V1).
2. feedback receptive fields and their orientation selectivity, of neurons in the primary visual cortex (V1), through inputs from the latero-medial visual cortex (LM).

## Repository structure 
Code developed using Python 3.11.13.

Required libraries: Numpy, Matplotlib, Scipy, tqdm, gwp, seaborn, pandas, starbars, open-cv.

To install, run: 
`pip install numpy matplotlib scipy tqdm gwp seaborn pandas starbars opencv-python`

Code files:

- simfeedfoward.py
- simfeedback.py
- generateffRF.py
- processffmodelresults.py
- processfbmodelresults.py

To run
