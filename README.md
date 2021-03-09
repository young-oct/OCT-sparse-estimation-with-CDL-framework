# Convolutional basis pursuit denoising with dictionary learning for deconvolution optical coherence tomography images



We show how a sparsity representation estimation approach that exploits the convolutional structure of optical coherence tomography (OCT) A-line data and that learns the axial point spread function from the image data offers an effective method of removing artefacts and stochastic noise from OCT images. Our method estimates the sparse representation by solving the convolutional basis pursuit denoising with dictionary learning (CPBN-DL) problem to learn the axial point spread function from imaging data and estimate the depth-resolved scattering amplitude on a line-by-line basis.  We demonstrate that this approach achieves effective artefact suppression and super-resolution while preserving anatomical structure and speckle texture in OCT images. The implementation is modified from 

[]: https://github.com/bwohlberg/sporco	"SPORCO"



![Picture1](/Users/youngwang/Desktop/Picture1.bmp)





## Dataset

We included middle ear, index finger (palmar view), index finger (side view), and onion slice A-line dataset. The OCT spectrogram data was collected using a previously described

[1]: https://www.osapublishing.org/boe/fulltext.cfm?uri=boe-7-11-4621&amp;id=352647	"long-range, wide-field swept-source optical coherence tomography with GPU accelerated digital lock-in Doppler vibrography for real-time, in vivo middle ear diagnostics"

, custom-built OCT system built around a 1550nm akinetic swept laser source (Insight Photonics SLE-101) with a sweep bandwidth of 40nm, axial resolution of 40µm in air, lateral resolution in air at the focus of 35µm and a nominal sweep rate of 100 kHz. 



Each dataset is store in Numpy array with a dimension of 10,240 x 330, that is 10,240 A-line with a length of 330 pixels.



## Usage

'oct_cdl.py' estimates the axial point spread function(PSF) from a subset of the A-line. 

Once the PSF is learned, 'oct_cbpdn' problem identifies a sparse representation of the image through convolutional basis pursuit denosing(CBPDN) framework. 



Following a successful download of the data. Line 36 in 'oct_cdl.py' and line 45, 46 in 'oct_cbpdn' should be changed to the correct file path. 

## Dependencies 

The following Python should be included to run the scripts

- []: https://github.com/bwohlberg/sporco	"SPORCO"

```
pip install git+https://github.com/bwohlberg/sporco
```

- []: https://scikit-image.org/	"scikit-image"

- []: https://docs.python.org/3/library/pickle.html	"pickle"

- []: https://pillow.readthedocs.io/en/stable/#	"Pillow"

- []: https://matplotlib.org/	"matplotlib"

- []: https://numpy.org/	"Numpy"

- []: https://pypi.org/project/pytictoc/	"pytictoc"

