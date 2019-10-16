#!/usr/bin/env python

import numpy as np
from geometry import Geometry
from timeit import default_timer as timer



def Bond_Dis(elem,xyz):
    for i, atom in enumerate(elem):
        for j in range(i):
            bond_len = np.linalg.norm(xyz[i]-xyz[j])  
            print (elem[i],elem[j],"Bond is",bond_len, "Bohr")

C2H4O = Geometry('C2H4O',
        coor='''
C  0.000000000000     0.000000000000     0.000000000000
C  0.000000000000     0.000000000000     2.845112131228
O  1.899115961744     0.000000000000     4.139062527233
H -1.894048308506     0.000000000000     3.747688672216
H  1.942500819960     0.000000000000    -0.701145981971
H -1.007295466862    -1.669971842687    -0.705916966833
H -1.007295466862     1.669971842687    -0.705916966833        
''',
                   charge=0,
                   multi=1,
                   basisname='sto-3g',
                   unit='bohr')

Bond_Dis(C2H4O.elem,C2H4O.xyz)


