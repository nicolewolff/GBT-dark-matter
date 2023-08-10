# GBT-dark-matter
## Axion dark matter search using Breakthrough Listen GBT data

[Original paper](https://iopscience.iop.org/article/10.3847/1538-4357/ac4d93) by Aya Keller et. al introducing the asymmetry analysis technique to search for dark matter using the Breakthrough Listen nearby star sample, and demonstrating the technique on a 100 MHz region of the L-Band

We are expanding this analysis to the complete L-Band, and eventually the complete dataset of L, S, C, and X-Band.

To adapt and/or run this analysis:
`git clone https://github.com/nicolewolff/GBT-dark-matter.git`

Contact me at new2128@columbia.edu for information about obtaining the datasets used for this analysis, or search for individual files [here](http://seti.berkeley.edu/opendata) to use to test a specific part of this analysis.

To convert h5/filterbank data products to NumPy .npy files (in Python):
```python
import blimpy as Waterfall
import numpy as np

wf = Waterfall()
data = wf.grab_data()
data = np.mean(data, axis=0)  # Average over the time axis
np.save('[filename].npy', data)
```

### Running the analysis

Open the file `config.ini`. Edit the directory paths accordingly. 
