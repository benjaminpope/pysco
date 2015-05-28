import numpy as np
import matplotlib.pyplot as plt
import whisky as pysco

from time import time

a = pysco.kpi('./geometry/medcross.txt',bsp_mat = 'sparse')

a.save_to_file('./geometry/medcrossmodel_sparse.pick')