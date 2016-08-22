from skimage import data, io, filters, transform
#from numpy import fft, log
from scipy import fftpack, misc
import matplotlib.pyplot as plt
import mpmath
from colorsys import hls_to_rgb
import Image
import numpy as np

def main():
    face = misc.face(gray=True)
    misc.imsave('face.png', face)
    face = misc.imread('DSC_1025.jpg', mode='RGB')
    # print(face)
    fourierImage = fftpack.fftn(face)**3
    shiftedFourierImage = fftpack.fftshift(fourierImage)
    mag = np.log(np.abs(shiftedFourierImage)+1)
    io.imshow(mag)
    io.show()

#image = data.camera()  # or any NumPy array!
#edges = filters.sobel(image)
#io.imshow(image)
#io.show()
#camera = data.camera()
# fourierImage = transform.frt2(camera)
# print(camera)
#fourierImage = log(abs(fft.fftshift(fft.fft2(camera, norm="ortho"))**2))
# fourierImage = fft.fft2(image)

    #mpmath.cplot(colorize(fourierImage), points=100000)

    #A = 1 / (fourierImage + 1j) ** 2 + 1 / (fourierImage - 2) ** 2
    #temp = colorize(A)
    #plt.imshow(fourierImage)
    #plt.show()
#misc.toimage(fourierImage).save('fft.png')
# misc.imsave('fft.png', fourierImage)
#plt.imshow(fourierImage)
#plt.show()
#io.imshow(fourierImage)
#io.show()
#io.imshow(camera)

# Image.open('img.jpg').convert('RGB').save('new.jpg')
# image = Image.open('img.jpg')
#image = image.convert('L')
#array = np.asarray(image)
# b = abs(np.fft.rfft2(array))
# j = Image.fromarray((b))
# j.save('img2.png')
def colorize(z):
    n,m, r = z.shape
    c = np.zeros((n,m,r))
    #c[np.isinf(z)] = (1.0, 1.0, 1.0)
    #c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = True
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0 - 1.0/(1.0+abs(z[idx])**0.3)
    c[idx] = [hls_to_rgb(a.all(), b.all(), 0.8) for a, b in zip(A, B)]
    return c

if __name__ == "__main__":
    main()