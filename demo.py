
import numpy as np
import time
from numpy.linalg import norm
from numpy.fft import fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import savemat, loadmat
import torch
from utils.utils_deblur import gauss_kernel, pad, crop
from p4ip import pnp_poisson
from models.network_dncnn import DnCNN as DnCNN
import torch.nn as nn

	
def load_net(MODEL_NAME):
	if MODEL_NAME == 'DnCNN':
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		net = DnCNN(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='R')
		model_file = 'model_zoo/dncnn_15.pth'
		net.load_state_dict(torch.load(model_file))
		net.eval()
		net.to(device)
	
	if MODEL_NAME == 'BM3D':
		return None

	return net 





np.random.seed(20)
H, W = 256, 256; K_IDX = 11
# Load test image
x = np.asarray(Image.open('data/Images/camera.png'))
x = x/255.0
if x.ndim > 2:
	x = np.mean(x,axis=2)
	# Reshape to form [N,N] image
x_im = Image.fromarray(x).resize((W,H))
x = np.asarray(x_im)


M =20;
"""
Choose kernel from list of kernels
"""
struct = loadmat('data/kernels_12.mat')
kernel_list = struct['kernels'][0]
kernel = kernel_list[K_IDX]
kernel = kernel/np.sum(kernel.ravel())
k_fft = fft2(pad(kernel, [H, W]))
A = lambda x : np.real(ifftshift(ifft2(fft2(x)*k_fft))) 


net = load_net('DnCNN')

y_n = M*A(x)
y = np.random.poisson(np.maximum(y_n,0))
y = np.asarray(y,dtype=np.float32)

# Poisson Plug-and-Play for Inverse Problems
s1=time.time()
x_list1 = pnp_poisson(y, kernel, M, net) 
x_pnp = x_list1[-1]
s2=time.time()


psnr =  -10*np.log10(np.mean((x_pnp-x)**2))
print('Conventional PnP')
print('Reconstruction PSNR: %0.3f, Elapsed Time: %0.3f s'%(psnr, s2-s1))

plt.subplot(1,3,1); plt.imshow(y, cmap='gray'); plt.axis('off')
plt.title('Noisy and Blurred Image')

plt.subplot(1,3,2); plt.imshow(x_pnp, cmap='gray'); plt.axis('off')
plt.title('Noisy and Blurred Image')

plt.subplot(1,3,3); plt.imshow(x, cmap='gray'); plt.axis('off')
plt.title('True Image')

plt.show()


plt.savefig('results/demo.png', bbox_inches='tight', pad_inches=0.05)
