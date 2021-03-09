# Convolutional basis pursuit denoising with dictionary learning for deconvolution optical coherence tomography images



We show how a sparsity representation estimation approach that exploits the convolutional structure of optical coherence tomography (OCT) A-line data and that learns the axial point spread function from the image data offers an effective method of removing artefacts and stochastic noise from OCT images. Our method estimates the sparse representation by solving the convolutional basis pursuit denoising with dictionary learning (CPBN-DL) problem to learn the axial point spread function from imaging data and estimate the depth-resolved scattering amplitude on a line-by-line basis.  We demonstrate that this approach achieves effective artefact suppression and super-resolution while preserving anatomical structure and speckle texture in OCT images. The implementation is modified from [SPORCO.](https://github.com/bwohlberg/sporco)

![Picture1](https://tva1.sinaimg.cn/large/008eGmZEly1godi32dqsyj30ki0bkgmz.jpg)

## Dataset

We included middle ear, index finger (palmar view), index finger (side view), and onion slice A-line dataset. The OCT spectrogram data was collected using a previously described in [[1]](https://www.osapublishing.org/boe/fulltext.cfm?uri=boe-7-11-4621&amp;id=352647) , custom-built OCT system built around a 1550nm akinetic swept laser source (Insight Photonics SLE-101) with a sweep bandwidth of 40nm, axial resolution of 40µm in air, lateral resolution in air at the focus of 35µm and a nominal sweep rate of 100 kHz. 

Each dataset is store in Numpy array with a dimension of 10,240 x 330, that is 10,240 A-line with a length of 330 pixels.

## Usage

`oct_cdl.py` estimates the axial point spread function(PSF) from a subset of the A-line. Once the PSF is learned, `oct_cdpdn.py` identifies a sparse representation of the image through convolutional basis pursuit denosing(CBPDN) framework. Following a successful download of the data. Line **36** in `oct_cdl.py` and line **45**, **46** in `oct_cdpdn.py` should be changed to the correct file path. 

## Dependencies 

The following Python should be included to run the scripts

- [SPORCO](https://github.com/bwohlberg/sporco)

```
pip install git+https://github.com/bwohlberg/sporco
```

- [scikit-image](https://scikit-image.org/ )
- [pickle](https://docs.python.org/3/library/pickle.html)
- [Pillow](https://pillow.readthedocs.io/en/stable/#)
- [pytictoc](https://pypi.org/project/pytictoc/)
- [Numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)

