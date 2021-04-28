# Convolutional basis pursuit denoising with dictionary learning for deconvolution optical coherence tomography images

We show how sparsity-regularized deconvolution with dictionary learning of the axial point spread function offers an effective method of removing artefacts and stochastic noise from OCT images. Our method estimates the sparse representation of the depth-resolved scattering amplitude of OCT A-lines on a line-by-line basis using the convolutional basis pursuit denoising with dictionary learning (CBPDN-DL) framework.  We employ a two two-pass process approach in which the first pass is used to segment the image and obtain a $l_1$ weighting mask that can be applied to the sparsity term in the optimization cost function within tissue-containing regions and the second pass to produce the final image sparse estimate. We show that sparsity-regularized deconvolution employing this weighting factor achieves good preservation of tissue structure while suppressing both noise and sidelobes and allows improved resolution of tissue structures. The implementation is modified from [SPORCO.](https://github.com/bwohlberg/sporco)

![image-20210428174146196](https://tva1.sinaimg.cn/large/008i3skNly1gq04n7n4twj318g0p01kx.jpg)

<sub>Figure 1. FFrom left to right: OCT images of a middle ear, index finger (palmar view), index finger (side view), and onion slice. The white arrow indicates the sidelobe artifacts caused by the PSF convolution.<sub>



## Dataset

We included middle ear, index finger (palmar view), index finger (side view), and onion slice A-line dataset. The OCT spectrogram data was collected using a previously described in [[1]](https://www.osapublishing.org/boe/fulltext.cfm?uri=boe-7-11-4621&amp;id=352647) , custom-built OCT system built around a 1550nm akinetic swept laser source (Insight Photonics SLE-101) with a sweep bandwidth of 40nm, axial resolution of 40µm in air, lateral resolution in air at the focus of 35µm and a nominal sweep rate of 100 kHz. 

Each dataset is store in Numpy array with a dimension of 10,240 x 330, that is 10,240 A-line with a length of 330 pixels. Each B-mode image is formed by sampling everything 20th line from its respective dataset. 

## Usage

`oct_cdl.py` estimates the axial point spread function(PSF) from a subset of the A-line. Once the PSF is learned, `weight_compare.py` ,`image_compare.py`,`lambda_compare.py`, `omega_compare.py` and `quality_compare.py`correspdonsd the **Figure 2, 3, 4, 5, 6** presented in the paper. 

## Dependencies 

The following Python should be included to run the scripts

- [SPORCO](https://github.com/bwohlberg/sporco)

```
pip install git+https://github.com/bwohlberg/sporco
```

- [scikit-image](https://scikit-image.org/ )
- [pickle](https://docs.python.org/3/library/pickle.html)
- [pytictoc](https://pypi.org/project/pytictoc/)
- [Numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)

