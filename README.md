# MACDA

Mcmc Assisted Coupler Designer Analyzer

Remote project by Dr. Ritoban Basu Thakur (Caltech) and Arjun Ghosh (Raman Research Institute).

This code is used to get the impedance values of any number of section symmetric 3dB Quadrature Hybrid Coupler. The basis of the code is a Markov Chain Monte Carlo based pipeline that estimates the impedance. emcee library is used for the MCMC implementation. There is also a analyzer function defined in the pipeline which can plot the S Parameters of any synmetric 3dB Quadrature Hybrid Coupler along with a tuner for fine tuning the impedance values.
