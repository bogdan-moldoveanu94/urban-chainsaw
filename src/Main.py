from skimage import data, io, filters, transform
from numpy import fft, log
import Image
import numpy as np


#image = data.camera()  # or any NumPy array!
#edges = filters.sobel(image)
#io.imshow(edges)
#io.show()
#camera = data.camera()
# fourierImage = transform.frt2(camera)
# print(camera)
#fourierImage = log(abs(fft.fftshift(fft.fft2(camera, norm="ortho"))**2))
#io.imshow(fourierImage)
#io.show()
#io.imshow(camera)

Image.open('img.jpg').convert('RGB').save('new.jpg')
image = Image.open('new.jpg')
image = image.convert('L')
array = np.asarray(image)
b = abs(np.fft.rfft2(array))
j = Image.fromarray((b))
j.save('img2.png')
