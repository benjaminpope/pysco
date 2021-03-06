#!/usr/bin/env python

''' -------------------------------------------------------
    This procedure generates a coordinates file for a hex
    pupil made of an arbitrary number of rings.
    Additional constraints on the location of spiders make
    it look like the your favorite telescope primary mirror
    ------------------------------------------------------- '''

import numpy as np, matplotlib.pyplot as plt
import time

nr     = 100            # rings within the pupil (should be ~> 50)
rmax   = 2.3918 # from optimization #5.093/2.  * 15.4/16.88        # outer diameter:      5.093 m
rmin   = rmax/2.11885 # 1.829/2. * 7.6/6.08         # central obstruction: 1.829 m
thick  = 0.2292*(rmax-rmin)# 0.83 *15.4/16.88/4.         # adopted spider thickness (meters)

srad = 0.33/3.          # segment "radius"
rad = np.sqrt(3)*srad  # radius of the first hex ring in meters

xs = np.array(())
ys = np.array(())

fig = plt.figure(0, figsize=(6,6))
plt.clf()
ax = plt.subplot(111)
circ1 = plt.Circle((0,0), rmax, facecolor='none', linewidth=1)
circ2 = plt.Circle((0,0), rmin, facecolor='none', linewidth=1)
ax.add_patch(circ1)
ax.add_patch(circ2)
#plt.clf()
ax.axis([-rmax,rmax, -rmax,rmax], aspect='equal')

for i in range(1-nr, nr, 1):
    for j in xrange(1-nr, nr, 1):
        x = srad * (i + 0.5 * j)
        y = j * np.sqrt(3)/2.*srad
        if (abs(i+j) < nr):
            xs = np.append(xs, x)
            ys = np.append(ys, y)

# modifications to match the actual telescope pupil (1): diameter constraints
# -----------------------------------------------------------------------
xx, yy = xs.copy(), ys.copy()        # temporary copies
xs, ys = np.array(()), np.array(())  # start from scratch again

for i in range(xx.size):
    thisrad = np.sqrt(xx[i]**2 + yy[i]**2)
    if (1.02 * rmin < thisrad < (0.98 * rmax)):# + 0.1*srad)):
        xs = np.append(xs, xx[i])
        ys = np.append(ys, yy[i])

# modifications to match the actual telescope pupil (2): spiders
# -----------------------------------------------------------
rm_spiders = True

if rm_spiders:
    xx, yy = xs.copy(), ys.copy()        # temporary copies
    xs, ys = np.array(()), np.array(())  # start from scratch again

    for i in range(xx.size):
        if (np.abs(xx[i]) >= thick/2.) and (np.abs(yy[i]) >= thick/2.):
            xs = np.append(xs, xx[i])
            ys = np.append(ys, yy[i])        


# plot segments
# -------------

rs = np.sqrt(xs**2+ys**2)
xs *= rmax/rs.max()
ys *= rmax/rs.max()

r0 = srad/np.sqrt(3)
th = 2*np.pi*np.arange(6)/6. + np.pi/6.

for i in range(xs.size):
    hx = xs[i] + r0 * np.cos(th)
    hy = ys[i] + r0 * np.sin(th)
    ax.fill(hx, hy, fc='none', linewidth=1)

ax.plot(xs, ys, 'r.')
plt.xlabel('m')
plt.ylabel('m')
plt.title('Hale 200-Inch Telescope Medium Cross Pupil Model')

np.savetxt("./geometry/medcrossmeas.txt", np.transpose((xs,ys)), 
           fmt='%12.9f')
print 'Maximum radius',np.max(np.sqrt(xs**2+ys**2))

print "--------------------------------------------------"
print "%d pupil sample points were included in the pupil " % xs.size
print "--------------------------------------------------"



plt.show()

