# Convolutional dictionary learning for deconvolution of optical coherence tomography images

We show that a novel image processing technique, convolutional dictionary learning (CDL), offers an effective method of removing sidelobe artefacts and stochastic noise from optical coherence tomography (OCT) images. Our method estimates the scattering amplitude of tissue on a line-by-line basis by estimating and deconvolving the 1-dimensional point spread function from OCT A-line data. We also present a method for employing a sparsity weighting mask to mitigate loss of speckle brightness within tissue-containing regions. With qualitative and quantitative analysis, we show that this approach is effective in suppressing sidelobe artefacts and background noise while preserving the intensity of tissue structures. It is particularly useful for emerging OCT applications where OCT images contain strong specular reflections at air-tissue boundaries.The implementation is modified from [SPORCO.](https://github.com/bwohlberg/sporco)

![1](https://tva1.sinaimg.cn/large/008i3skNgy1greyz4w0dwj318g0p0wom.jpg)<sub>Figure 1. From left to right: OCT images of a middle ear, index finger (palmar view), index finger (side view), and onion slice. The top row shows the reference image with only standard processing applied while the bottom row shows the sparse vector images obtained from two-pass sparsity-regularized deconvolution. The white arrow indicates the sidelobe artifacts caused by the PSF. <sub>



## Dataset

We included middle ear, index finger (palmar view), index finger (side view), and onion slice A-line dataset. The OCT spectrogram data was collected using a previously described in [[1]](https://www.osapublishing.org/boe/fulltext.cfm?uri=boe-7-11-4621&amp;id=352647) , custom-built OCT system built around a 1550nm akinetic swept laser source (Insight Photonics SLE-101) with a sweep bandwidth of 40nm, axial resolution of 40µm in air, lateral resolution in air at the focus of 35µm and a nominal sweep rate of 100 kHz. 

Each A-line dataset is stored as Numpy array with a dimension of 10,240 x 330, that is 10,240 A-line with a length of 330 pixels. A-line data is acuqired by a standard processing step described in  **Figure 1**, where a <u>**Hann**</u> window with the same lenght of interferogram was applied pior to inverse Fourier transformation(IFFT).

Each B-mode image is formed by sampling everything 20th line from its respective dataset. 

-  `ear` : OCT middle ear A-line data, stored as Numpy array with a dimension of 10,240 x 330. 
-  `finger` : OCT index finger(palmar view) A-line data, stored as Numpy array with a dimension of 10,240 x 330. 
-  `finger(raw)` : Raw interferogram of  OCT index finger(palmar view), stored in  `.npz`  file format and accessed with key  `arr_1` . The raw interferogram is saved as Numpy array with a dimension of 25,000 x 1460. That is 25,000 interferogram with a length of 1460(data valid points) pixels. 
-  `nail` : OCT index finger(side view) A-line data, stored as Numpy array with a dimension of 10,240 x 330. 
-  `onion` : OCT onion slice A-line data, stored as Numpy array with a dimension of 10,240 x 330. 

The  `PSF` folder contains the PSFs learned from each dataset, respectively. 



## Usage

- `oct_cdl.py` estimates the axial point spread function(PSF) from a subset of the A-line
- `weight_compare.py` illustrates result of performing a second pass of the CSC optimization with this weighting mask, and corresponds to **Figure 2** presented in the paper. 
- `image_compare.py` demonstrates the general appliclity of the proposed method with various datasets, and corresponds to  **Figure 3** presented in the paper. 
- `lambda_compare.py` shows the effects of varying $\lambda$ from 0.01 to 0.2 with a fixed $W=0.1$ for the middle ear image,  and corresponds to **Figure 4** presented in the paper. 
-  `omega_compare.py`  depicts the effects of varying  $W$ from 0.1 to 1 with a fixed  $\lambda=0.5$ for the middle ear image,  and corresponds to **Figure 5** presented in the paper. 
- `lambda_gCNR.py` assesses the performance of the proposed method with a newly established image quality metric generalized contrast-to-noise ratio([gCNR](https://ieeexplore.ieee.org/document/8580101)), and corresponds **Figure 6** presented in the paper. 
- `lambda_gCNR.py`  compares the proposed method to spectral windowing with two frequently used window functions(Gassuan amd Hann),  and corresponds **Figure 7** presented in the paper. 

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

