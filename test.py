import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio


import numpy as np


def convolve(f, kernel):
    m, n = kernel.shape

    y, x = f.shape
    y = y - m + 1
    x = x - m + 1
    g = np.zeros((y,x))
    for i in range(y):
        for j in range(x):
            g[i][j] = np.sum(f[i:i+m, j:j+n]*kernel)
    
    return g

c = imageio.imread('noisy.png')
im = c.copy()
im = im /255.
gaus = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])
gaus = gaus * 1/273

#print (naive_convolve(im[0],gaus))

imc = None
r = convolve(im[:,:,0], gaus)
g = convolve(im[:,:,1], gaus)
b = convolve(im[:,:,2], gaus)
print(r.shape)
print(g.shape)
print(b.shape)
im_out_test = np.dstack([r, g, b])
print(im_out_test.shape)
#r = scipy.signal.convolve2d(im[:,:,0], gaus, mode='same')
#g = scipy.signal.convolve2d(im[:,:,1], gaus, mode='same')
#b = scipy.signal.convolve2d(im[:,:,2], gaus, mode='same')
#print(r.shape)
#print(g.shape)
#print(b.shape)
#im_out = np.dstack([r, g, b])
#print(im_out.shape)

plt.subplot(2,1,1)
plt.imshow(im)  
plt.subplot(2,1,2)
plt.imshow(im_out_test)  
##plt.subplot(2,1,3)
##plt.imshow(im_out)
plt.show()

"""


c = imageio.imread('noisy.png')
im = c.copy()
im = im /255.
#im.setflags(write=1)
print (im,"\naaaaaaaaaaaa")
#plt.subplot(2,1,1)
#plt.imshow(im)
#plt.show()
# normalise to 0-1, it's easier to work in float space
#im = c.copy()
# make some kind of kernel, there are many ways to do this...
t = 1 - np.abs(np.linspace(-1, 1, 21))
kernel = t.reshape(21, 1) * t.reshape(1, 21)
kernel /= kernel.sum()   # kernel should sum to 1!  :) 

# convolve 2d the kernel with each channel
print("bbbbbbbbbbbbbbb=", im[:,:,0][0])
r = np.convolve(im[:,:,0][0], kernel, mode='same')
g = np.convolve(im[:,:,1], kernel, mode='same')
b = np.convolve(im[:,:,2], kernel, mode='same')
#print("\n\n\n\n",im)
# stack the channels back into a 8-bit colour depth image and plot it
im_out = np.dstack([r, g, b])
im_out = (im_out * 255).astype(np.uint8) 
print("\n\n\n\n",im_out)
plt.subplot(2,1,1)
plt.imshow(im)  
plt.subplot(2,1,2)
plt.imshow(im_out)
plt.show()"""