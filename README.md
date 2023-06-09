# Modeling-Visual-Perception

(1) Create a 1920 x 1080 pixel spectral image (400-700nm sampled every 1nm) of a spectrum
spread uniformly across the image. Use a 5nm triangular bandpass to create your spectral
samples. Create a second image, from the first, with 5nm increments. Sample several pixels
from each image and turn in their spectral plots. 

(2) Render your image (you can use the 5nm version from here on if you prefer) to sRGB and
display it. Save as a .tiff file. 

(3) Create a mosaic of about 100 circular cone apertures that pack tightly into a 1920x1080
image. Label them appropriately as L, M, and S. 

(4) Compute the LMS responses to your spectral image in each cone aperture and render the
cone sampled image in grayscale for response level and color coded to also show cone type.

(5) Simulate center-surround in luminance (on, off) and color (R-G, G-R, Y-B, B-Y) using only
the center cone and its nearest neighbors. You will have six results. Render grayscale and color
coded output images 

(6) Compute CIECAM02 lightness and brightness of your spectrum for 3 stimulus luminance
levels. State the parameters used. Render the lightness and brightness images and compare
with a luminance image. 

(7) Compute CIECAM02 colorfulness and saturation of your spectrum for 3 stimulus luminance
levels. State the parameters used. Render the colorfulness and saturation images in grayscale
and figure out a useful way to render in full color.

(8) Start with spectral image (your spectrum or other) and render it for humans, for one chosen
animal, and for a protanope.
