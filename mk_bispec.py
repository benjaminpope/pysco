import numpy as np
import matplotlib.pyplot as plt
import whisky as pysco

from time import time

a = pysco.kpi('./geometry/old_med_cross.txt', bsp_mat = 'full',verbose=True)

a.save_to_file('./geometry/oldmedcrossmodel.pick')

plt.clf()
plt.hist(np.log10(a.bsp_s))
plt.xlabel('log10 (singular values)')
plt.ylabel('N')
plt.title('SVD Histogram')
plt.savefig('svdhist.png')