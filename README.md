# NA-MEMD for Python


## Introduction
Python implementation of Noise-Assisted Multivariate Empirical Mode Decomposition (NA-MEMD). This repository contains functions for noise generation, Multivariate Empirical Mode Decomposition (MEMD), visualization, etc.  

Multivariate Empirical Mode Decomposition (MEMD) is an extension of the traditional Empirical Mode Decomposition (EMD) method, which is used to decompose non-linear and non-stationary signals into simpler oscillatory components known as Intrinsic Mode Functions (IMFs). The key feature of MEMD is its ability to handle multivariate signals, meaning it can simultaneously decompose multiple related signals (or different dimensions of a signal) while ensuring that the decomposition is consistent across all channels.

Noise-Assisted Multivariate Empirical Mode Decomposition (NA-MEMD) is an enhancement of MEMD that incorporates noise into the decomposition process. The noise is added to make the decomposition more robust and to help separate closely spaced frequency components (mode mixing) that traditional MEMD might struggle to distinguish. The idea is that the added noise helps to explore the signal more effectively and results in more accurate decomposition.
  
__Key Benefits of Repository__ : 
- Customizable Noise Generation :
  - The repository provides flexible options for noise generation, allowing for customization based on the type of data or the problem being addressed.
- Multivariate Decomposition :
  - Decomposes n-dimensional signals, not limited to bivariate or quadrivariate, making it applicable to complex, high-dimensional data.
- Performance Optimization :
  - The implementation is significantly faster than existing models, making it practical for large-scale and real-time data applications.
- Visualization :
  - Provides tools for basic visualization, enabling users to easily examine the results of the decomposition and compare them with the original signal.


## Dependencies 
- NumPy
- SciPy  
- sys
- math
- matplotlib

```bash
# sys and math are part of Python Standard Libraries
pip install numpy scipy matplotlib
```


## General Functions and Usage  
```python
signal = np.random.randn(5, 1000)
memd_imfs = memd(signal)  # traditional memd signal processing
na_memd_imfs = na_memd(signal)  # na_memd processing with noise assistance
add_noise = add_noise(signal)  # add noise only to signal without EMD processing
```
For additional functionalities and usage explanations, please reference the example.py file.


## Acknowledgements
Several existing packages and repositories were referenced in the creation of this library. All credit goes to these authors for their contributions to the field.

* https://github.com/AaronLi43/ginkgo_glasgow [1]
* https://github.com/laszukdawid/PyEMD/tree/master
* https://github.com/mariogrune/MEMD-Python-/tree/master [2]
* https://github.com/STherese/NA-MEMD-for-EEG


## Citations
[1] Y. Zhang, G. Wang, Z. Li, et al., "Matlab Open Source Code: Noise-Assisted Multivariate Empirical Mode Decomposition Based Causal Decomposition for Causality Inference of Bivariate Time Series," Front. Neuroinform., vol. 16, Art. no. 851645, Jun. 2022. doi: 10.3389/fninf.2022.851645  [LINK](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9243260/)  
[2] “Research | Empirical Mode Decomposition (EMD), Multivariate EMD, Matlab code and data sources ∴ Dr. Danilo P. Mandic,” www.commsp.ee.ic.ac.uk. https://www.commsp.ee.ic.ac.uk/~mandic/research/emd.htm (accessed Jul. 09, 2024)  [LINK](https://www.commsp.ee.ic.ac.uk/~mandic/research/emd.htm)

## Other
Plans for future updates, such as performance improvements and additional features, are listed within the example.py file as comments.

## Contact
This is a work in progress. If you encounter any issues or have questions, feel free to reach out via email at zhue@ufl.edu.


‌



