#!/usr/bin/env python

import numpy as np
import parameter
from fock import RHF_direct_Fock
from geometry import Geometry
from timeit import default_timer as timer

#def Bond_Dis(elem,xyz):
#    for i, atom in enumerate(elem):
#        for j in range(i):
#            bond_len = np.linalg.norm(xyz[i]-xyz[j])  
#            print (elem[i],elem[j],"Bond is",bond_len, "Bohr")

def AT_x_B_x_A(A,B):
    
    tmp=np.matmul(A.transpose(),B)
    C  =np.matmul(tmp,A)
    return C
    
def A_x_B_x_AT(A,B):

    tmp=np.matmul(B,A.transpose())
    C  =np.matmul(A,tmp)
    return C

def RHF_Density(nel, C):
    
    col = C.shape[1]
    nocc=nel//2
    nvir=col-nocc
    tmp = np.delete(C, np.linspace(nocc, col-1, num=nvir, dtype='int'), 1)
    density = np.matmul(tmp, tmp.transpose())    

    return density

def RHF_Energy(density, hcore, fock, enuc):

    tmp = np.add(hcore,fock)
    energy =  (density*tmp).sum()
    return energy+enuc

def RHF_Fock(eri,density,hcore):
    nbas = hcore.shape[0] 
    nbas2= nbas*nbas

    start = timer()
    tmp1 = 2.0*np.tensordot(density,eri,axes=([0,1],[0,1]))
    #tmp1 = 2.0*np.einsum('ij,ijkl->kl', density, eri, optimize=True);
    end = timer()
    print("Fock matrix step 1 takes %f sec." % (end-start))    
    start = timer()
    #tmp2 = np.tensordot(density,eri,axes=([0,1],[0,2]))
    tmp2 = np.einsum('ij,ikjl->kl', density, eri, optimize=True);
    end = timer()
    print("Fock matrix step 2 takes %f sec." % (end-start))
    fock = np.add(hcore,tmp1)
    fock = np.subtract(fock, tmp2)
    return fock

CH4 = Geometry('CH4',
               coor='''
                       C  -0.000000000000   0.000000000000   0.000000000000
                       H   1.183771681898  -1.183771681898  -1.183771681898
                       H   1.183771681898   1.183771681898   1.183771681898
                       H  -1.183771681898   1.183771681898  -1.183771681898
                       H  -1.183771681898  -1.183771681898   1.183771681898
                   ''',
                   charge=0,
                   multi=1,
                   basisname='cc-pvdz',
                   unit='bohr')

s = CH4._get_S()
#print(s)
h = CH4._get_HCore()
#print(h)

eri = CH4._get_ERI_L()
#print(eri)

start = timer()
E,U=np.linalg.eigh(s)
S21 = A_x_B_x_AT(U,np.diag(E**(-0.5)))
#print(S21)

fock = AT_x_B_x_A(S21,h)
E,U=np.linalg.eigh(fock)
c=np.matmul(S21, U)
d = RHF_Density(CH4.nel, c)
end = timer()
print("Initial density matrix takes %f sec." % (end-start))

e0=0.0
for x in range(parameter._MAXITER):

    start = timer()
    #f = RHF_direct_Fock(CH4,h,d)
    #print(f)
    f = RHF_Fock(eri,d,h)  
    #print(f)
    e = RHF_Energy(d, h, f, CH4.enuc)
        
    fock = AT_x_B_x_A(S21,f)
    E,U=np.linalg.eigh(fock)
    c=np.matmul(S21, U)
    d = RHF_Density(CH4.nel, c)
    if (abs(e-e0)<1.0e-7):
        break
    end  = timer()
    print ("Iter %d E=  %lf  Delta E=  %lf Time= %f sec." % (x, e, e-e0, end-start) )
    e0 =e 


