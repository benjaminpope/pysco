#!/usr/bin/env python

import numpy as np, matplotlib.pyplot as plt

radius = 5.092/2.
obst = 1.829/2.

nb_sub = 15  # number of sub-apertures accross one pupil-diameter
obstr  = obst/radius # size of central obstruction (33 %, from tiny tim data)

x = np.linspace(0, 2, nb_sub) - 1.0
y = np.linspace(0, 2, nb_sub) - 1.0

coords = np.zeros((0,2))

for i in range(nb_sub):
    for j in range(nb_sub):
        if ((obstr < np.hypot(x[i],y[j]) <= 0.9) and
            (abs((np.arctan(y[j]/x[i]))) > 0.05) and
            (abs((np.arctan(y[j]/x[i]) -np.pi/2)) > 0.05)
            ):
            coords=np.append(coords, [[x[i], y[j]]], axis=0)

coords *= radius # diameter of Palomar is 5.1m

plt.clf()

f1 = plt.subplot(111)

cir1 = plt.Circle((0.0,0.0), radius=radius, fc='gray', fill=True, linewidth=1)
cir2 = plt.Circle((0.0,0.0), radius=obst, color='white')
cir3 = plt.Circle((0.0,0.0), radius=1.7/1.2*radius, ec='white', fill=False, linewidth=130)
f1.plot(coords[:,0], coords[:,1], 'ro')

f1.plot([-radius,radius], 
         [0,0],
         color='white', linewidth=20)
f1.axis('equal')

f1.plot([0,0], 
         [radius,-radius],
         color='white', linewidth=20)

f1.axis('equal')

f1.add_patch(cir1)
f1.add_patch(cir3)
f1.add_patch(cir2)

plt.title('Palomar pupil model')
plt.axis([-3.0,3.0,-3.0,3.0], aspect='auto')

np.savetxt("geometry/palomar.txt", np.transpose((coords[:,0], coords[:,1])), 
           fmt='%6.3f')
plt.show()
