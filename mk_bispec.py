import numpy as np
import matplotlib.pyplot as plt
import whisky as pysco

from time import time

a = pysco.kpi('./geometry/medcross.txt', bsp_mat = 'full',verbose=True)

a.save_to_file('./geometry/medcrossmodel_new.pick')

plt.clf()
plt.hist(np.log10(a.bsp_s))
plt.xlabel('log10 (singular values)')
plt.ylabel('N')
plt.title('SVD Histogram')
plt.savefig('svdhist.png')