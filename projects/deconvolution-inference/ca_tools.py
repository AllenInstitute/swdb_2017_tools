from functions import deconvolve # OASIS import 
import numpy as np

def ca_deconvolution(ddf_trace): 
	""" perform calcium image deconvolution 
	
	This function performs several calcium image 
	deconvolution approaches. Deconvolutions currently 
	supported: 
	
	1. OASIS (https://github.com/j-friedrich/OASIS)
	2. Event detection script from Peter

	
	input: 
		ddf_trace: a 1d-numpy array of length n (the number of 
		time steps in the calcium trace)
		
	output: 
		a dictionary whose keys are the deconvolution method 
		used and values are a 1d-numpy array of length n with
		the estimated spikes 
	
	TODO:
	
	Add functionality for the following methods
	
	3. AR-FPOP
	4. ML Spike 
	5. One of the supervised methods? 	
	
	"""
	
	out = {}
	
	# Method 1 OASIS (https://github.com/j-friedrich/OASIS)
	c, s, b, g, lam = deconvolve(np.double(ddf_trace), penalty=1)
	out['OASIS'] = s
	
	return out


	