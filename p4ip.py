import numpy as np
import torch 

import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, ifftshift
from scipy.optimize import fmin_l_bfgs_b as l_bfgs
from torch.autograd import Variable
from utils.utils_torch import conv_fft, img_to_tens, scalar_to_tens
from utils.utils_pnp import l2_deconv, psf_to_otf, dncnn_wrapper, x_subp

def pnp_poisson(y, kernel, M, net):

	# ADJUSTABLE PARAMETERS
	lambda_r = 0.01; rho0 = 500
	MAX_ITERS = 100	
	TOL = 1e-3
	verbose = True
	

	H, W = np.shape(y)
	k_pad, k_fft = psf_to_otf(kernel, [H,W])

	A = lambda x : np.real(ifft2(fft2(x)*k_fft))
	At = lambda x : np.real(ifft2(fft2(x)*np.conj(k_fft))) 
	A_m = lambda x : M*A(x); At_m = lambda x : M*At(x) 
	
	rho = rho0
	params = {
	'y': y,
	'A': A_m,
	'At': At_m,
	'rho': rho,
	'x0': np.zeros([H*W],  dtype=np.float32)
	}
	# Initialize x through Wiener deconvolution
	x = np.clip(l2_deconv(y/M, k_fft, 1/M),0,1)
	v = x.copy()
	u = np.zeros([H,W], dtype=np.float32)
	delta = 0.0
	x_list = []
	for iters in range(MAX_ITERS):
		x_prev, v_prev, u_prev = x, v, u
		
		# L-BFGS solver for data subproblem
		xhat = np.reshape(v -  u, [H*W])
		params['rho'] = rho; params['x0'] = xhat
		x, f, dict_opt  = l_bfgs(func = x_subp, x0 = xhat, fprime=None, args=(params,), approx_grad=0)
		x = np.reshape(x,[H,W])
		
		# Denoising step
		vhat = x + u
		sigma = np.sqrt(lambda_r/rho)
		v = dncnn_wrapper(vhat, sigma, net)
		
		# Scaled lagrangian update
		u = u + x - v


		rel_diff_v = np.linalg.norm(v-v_prev,'fro')/np.sqrt(H*W)
		rel_diff_x = np.linalg.norm(x-x_prev,'fro')/np.sqrt(H*W)
		rel_diff_u = np.linalg.norm(u-u_prev,'fro')/np.sqrt(H*W)
		delta_prev = delta
		delta = 0.33*(rel_diff_x + rel_diff_v + rel_diff_u)
		x_list.append(x)
		if verbose:	print('Iteration: ', (iters+1))
		if delta > 0.99*delta_prev:
			rho *= 1.01
			if verbose:	print('Rho updated to %0.3f'%(rho))
		else:
			if verbose:	print('rho constant at %0.3f'%(rho))
		if verbose:	print('Relative Differences: %0.4f, %0.4f, %0.4f'%(rel_diff_x, rel_diff_v, rel_diff_u))
		if delta < TOL:
			break
	return x_list


