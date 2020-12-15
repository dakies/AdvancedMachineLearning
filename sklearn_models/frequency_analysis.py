import numpy as np 
import pandas as pd 
import sklearn 
from scipy import signal 
import biosppy 

# frequency bands considering normalized sampling frequency
freq_bands_general = {'delta':[0,0.04],
					  'theta':[0.04, 0.08],
					  'alpha_low':[0.08, 0.1],
					  'alpha_high':[0.1, 0.13],
					  'beta':[0.13, 0.25],
					  'gamma':[0.25, 0.5]}

def get_frequency_features(sig, signal_type='eeg'):
	# remember to apply this function after signal filtering

	features = {}

	# Energy properties
	freqs, psd = signal.welch(sig)

	lower = np.where(freqs>=freq_bands_general["delta"][0])
	upper = np.where(freqs<freq_bands_general["delta"][1])
	indexes = np.intersect1d(lower,upper)
	features["delta"] = np.trapz(y=psd[indexes], x=freqs[indexes])

	lower = np.where(freqs >= freq_bands_general["theta"][0])
	upper = np.where(freqs < freq_bands_general["theta"][1])
	indexes = np.intersect1d(lower, upper)
	features["theta"] = np.trapz(y=psd[indexes], x=freqs[indexes])

	lower = np.where(freqs >= freq_bands_general["alpha_low"][0])
	upper = np.where(freqs < freq_bands_general["alpha_low"][1])
	indexes = np.intersect1d(lower, upper)
	features["alpha_low"] = np.trapz(y=psd[indexes], x=freqs[indexes])

	lower = np.where(freqs >= freq_bands_general["alpha_high"][0])
	upper = np.where(freqs < freq_bands_general["alpha_high"][1])
	indexes = np.intersect1d(lower, upper)
	features["alpha_high"] = np.trapz(y=psd[indexes], x=freqs[indexes])

	lower = np.where(freqs >= freq_bands_general["beta"][0])
	upper = np.where(freqs < freq_bands_general["beta"][1])
	indexes = np.intersect1d(lower, upper)
	features["beta"] = np.trapz(y=psd[indexes], x=freqs[indexes])

	lower = np.where(freqs >= freq_bands_general["gamma"][0])
	upper = np.where(freqs < freq_bands_general["gamma"][1])
	indexes = np.intersect1d(lower, upper)
	features["gamma"] = np.trapz(y=psd[indexes], x=freqs[indexes])

	features["Total"] = np.trapz(y=psd,x=freqs)

	#Some additional features to test
	#features["extra1"]=features["LF"]/(features["LF"]+features["HF"])
	#features["extra2"]=features["HF"]/(features["LF"]+features["HF"])
	#features["extra3"]=features["LF"]/features["HF"]
	#features["extra4"]=features["HF"]/features["Total"]
	#features["extra5"]=features["HF"]/features["Total"]

	# Autocorrelation properties
	autocorr = np.correlate(sig,sig,"full")
	features["corr_max"]=np.max(autocorr)
	features["corr_min"]=np.min(autocorr)

	sig_bar = np.mean(sig)

	N = len(freqs)
	normalization = np.sum(np.square(sig-sig_bar))
	len1 = N - int(N/4)
	r1 = np.sum((sig[:len1]-sig_bar)*(sig[len1:2*len1]-sig_bar)) / normalization
	len2 = N - int(N/2)
	r2 = np.sum((sig[:len2]-sig_bar)*(sig[len2:2*len2]-sig_bar)) / normalization
	len3 = N - int(3*N/4)
	r3 = np.sum((sig[:len3]-sig_bar)*(sig[len3:2*len3]-sig_bar)) / normalization
	#r1 = np.sum((sig[:int(N-3*N/4)]-sig_bar)*(sig[int(3*N/4):int(N-3*N/4)]-sig_bar))/np.sum(np.square(sig-sig_bar))
	#r2 = np.sum((sig[:int(N-2*N/4)]-sig_bar)*(sig[int(2*N/4):int(N-2*N/4)]-sig_bar))/np.sum(np.square(sig-sig_bar))
	#r3 = np.sum((sig[:int(N-1*N/4)]-sig_bar)*(sig[int(N/4):int(N-N/4)]-sig_bar))/np.sum(np.square(sig-sig_bar))

	features["corr1"]=r1
	features["corr2"]=r2
	features["corr3"]=r3

	return features











