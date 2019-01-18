#!/usr/bin/env python

import copy
import sys
import basis_set_exchange as bse
from  basis_set_exchange import lut
import parameter
import numpy as np
from scipy.special import factorial2 as fact2
from one_int import shell_to_basis
from one_int import S, T, V, ERI, S_C, T_C, V_C, ERI_C, ERI_L 
from timeit import default_timer as timer

class Geometry:
    """
    A class handling geometry objects
 
    """
    def __init__(self, name=None, coor=None, **kwargs):
        self.name   = name
        self.elem   = []
        self.proton = []
        self.xyz    = []
        self.basis  = []
        self.nbs    = 0
        self.charge = int(kwargs.pop('charge',parameter._DEFAULT_CHARGE))
        self.multi  = int(kwargs.pop('multi', parameter._DEFAULT_MULTI))
        self.bsname = kwargs.pop('basisname', parameter._DEFAULT_BASIS)

        start = timer()
        if coor: self._read_atoms(coor)
        end = timer()
        print("read atom",end - start) # Time in seconds        

        start = timer()
        self._assign_basis()
        end = timer()
        print("assign basis", end - start) # Time in seconds

        self._get_S()
        self._get_S_C()


        self._get_T()
        self._get_T_C()

        self._get_V()
        self._get_V_C()

        self._get_ERI_L()
        self._get_ERI_C()

        return

    def _read_atoms(self, coor):
        atoms = coor.split()
        #print(atoms)
        natom=len(atoms)
        for i in range(0, natom, 4):
            self.elem.append(atoms[i])
            sym, number, name = lut.element_data_from_sym(atoms[i])
            self.proton.append(float(number))
            self.xyz.append(parameter._BOHR*np.array([atoms[i+1], atoms[i+2], atoms[i+3]], dtype=np.float64))

        print("Number  Elements  (No.)         X                Y               Z")
        for id, atom in enumerate(self.elem):
            print("%6d  %5s  %6d  %16.10f %16.10f %16.10f" % (id,self.elem[id],self.proton[id],self.xyz[id][0],self.xyz[id][1],self.xyz[id][2]))

    def _get_all_shells(self, m):

        momentum_string ={
           0: [[0,0,0]],
           1: [[1,0,0], [0,1,0], [0,0,1]],
           2: [[2,0,0], [1,1,0], [1,0,1], [0,2,0], [0,1,1], [0,0,2]],
           3: [[3,0,0], [2,1,0], [2,0,1], [1,2,0], [1,1,1], [1,0,2], [0,3,0], [0,2,1], [0,1,2], [0,0,3]]
        }
        return (momentum_string[m])

    def _assign_basis(self):

        for id, elem in enumerate(self.elem):

            sym, number, name = lut.element_data_from_sym(elem)
            bs_str = bse.get_basis(self.bsname, elements=sym, header=False)

            for k, el in bs_str['basis_set_elements'].items():

                if not 'element_electron_shells' in el:
                    continue

                for sh in el['element_electron_shells']:
                    exponents = sh['shell_exponents']
                    # transpose of the coefficient matrix
                    coeff_t = sh['shell_coefficients']
                    am = sh['shell_angular_momentum']
                    # loop over all momentum
                    for counter, value in enumerate(am):
                        # shell loop over each momentum
                        #for shell in self._get_all_shells(value):
                        #assign XYZ for each shell and scale with BOHR
                        #xyz = np.array(self.xyz[id], dtype='f')
                        xyz = np.array(self.xyz[id]) 
                        #assign angular_momentum for each shell
                        #mom = np.array(shell, dtype=np.int32)
                        mom = int(value)
                        #assign exponents for each shell
                        #exp = np.array(list(map(float, exponents)),dtype='f')
                        exp = np.array(list(map(float, exponents)))
                        #assign coefficients for each shell
                        #coef= np.array(list(map(float, coeff_t[counter])),dtype='f')
                        coef= np.array(list(map(float, coeff_t[counter])))
                        self.basis.append(Basis(coef, exp, mom, xyz)) 
                        self.nbs +=shell_to_basis(mom)                     

    def _get_S(self):
        s = np.zeros(shape=(self.nbs,self.nbs));
        print(self.nbs)
        time=0.0
        offi=0
        for i, x in enumerate(self.basis):
            offj=0
            for j in range(i+1):
                start = timer()
                tmp=S(self.basis[i].c, self.basis[i].p, self.basis[i].n, self.basis[i].e, self.basis[i].m, self.basis[i].xyz,
                      self.basis[j].c, self.basis[j].p, self.basis[j].n, self.basis[j].e, self.basis[j].m, self.basis[j].xyz)
                s[offi:(offi+tmp.shape[0]), offj:(offj+tmp.shape[1])]=tmp
                if (i!=j) :
                    tmp2=tmp.transpose()
                    s[offj:(offj+tmp2.shape[0]), offi:(offi+tmp2.shape[1])]=tmp2
                offj+=shell_to_basis(self.basis[j].m)
                end = timer()
                time+=end-start
            offi+=shell_to_basis(self.basis[i].m) 
        print("S construction from Python takes %f sec." % time) # Time in seconds
        print(s)

    def _get_S_C(self):
        s = np.zeros(shape=(self.nbs,self.nbs));
        time=0.0
        offi=0
        for i, x in enumerate(self.basis):
            offj=0
            for j in range(i+1):
                start = timer()
                tmp=S_C(self.basis[i].c, self.basis[i].p, self.basis[i].n, self.basis[i].e, self.basis[i].m, self.basis[i].xyz,
                        self.basis[j].c, self.basis[j].p, self.basis[j].n, self.basis[j].e, self.basis[j].m, self.basis[j].xyz)
                s[offi:(offi+tmp.shape[0]), offj:(offj+tmp.shape[1])]=tmp
                if (i!=j) :
                    tmp2=tmp.transpose()
                    s[offj:(offj+tmp2.shape[0]), offi:(offi+tmp2.shape[1])]=tmp2
                offj+=shell_to_basis(self.basis[j].m)
                end = timer()
                time+=end-start
            offi+=shell_to_basis(self.basis[i].m)
        print("S construction from C/C++ takes %f sec" % time) # Time in seconds
        print(s)

    def _get_T(self):
        t = np.zeros(shape=(self.nbs,self.nbs));
        time=0.0
        offi=0
        for i, x in enumerate(self.basis):
            offj=0
            for j in range(i+1):
                start = timer()
                tmp=T(self.basis[i].c, self.basis[i].p, self.basis[i].n, self.basis[i].e, self.basis[i].m, self.basis[i].xyz,
                      self.basis[j].c, self.basis[j].p, self.basis[j].n, self.basis[j].e, self.basis[j].m, self.basis[j].xyz)
                t[offi:(offi+tmp.shape[0]), offj:(offj+tmp.shape[1])]=tmp
                if (i!=j) :
                    tmp2=tmp.transpose()
                    t[offj:(offj+tmp2.shape[0]), offi:(offi+tmp2.shape[1])]=tmp2
                offj+=shell_to_basis(self.basis[j].m)
                end = timer()
                time+=end-start
            offi+=shell_to_basis(self.basis[i].m)
        print("T construction from Python takes %f sec." % time) # Time in seconds
        print(t)

    def _get_T_C(self):
        t = np.zeros(shape=(self.nbs,self.nbs));
        print(self.nbs)
        time=0.0
        offi=0
        for i, x in enumerate(self.basis):
            offj=0
            for j in range(i+1):
                start = timer()
                tmp=T_C(self.basis[i].c, self.basis[i].p, self.basis[i].n, self.basis[i].e, self.basis[i].m, self.basis[i].xyz,
                        self.basis[j].c, self.basis[j].p, self.basis[j].n, self.basis[j].e, self.basis[j].m, self.basis[j].xyz)
                t[offi:(offi+tmp.shape[0]), offj:(offj+tmp.shape[1])]=tmp
                if (i!=j) :
                    tmp2=tmp.transpose()
                    t[offj:(offj+tmp2.shape[0]), offi:(offi+tmp2.shape[1])]=tmp2
                offj+=shell_to_basis(self.basis[j].m)
                end = timer()
                time+=end-start
            offi+=shell_to_basis(self.basis[i].m)
        print("T construction from C/C++ takes %f sec" % time) # Time in seconds
        print(t)


    def _get_V(self):
        v = np.zeros(shape=(self.nbs,self.nbs));
        xyz_all    = np.array(self.xyz)
        proton_all = np.array(self.proton)
        time=0.0
        offi=0
        for i, x in enumerate(self.basis):
            offj=0
            for j in range(i+1):
                start = timer()
                tmp=V(self.basis[i].c, self.basis[i].p, self.basis[i].n, self.basis[i].e, self.basis[i].m, self.basis[i].xyz,
                      self.basis[j].c, self.basis[j].p, self.basis[j].n, self.basis[j].e, self.basis[j].m, self.basis[j].xyz, xyz_all, proton_all)
                v[offi:(offi+tmp.shape[0]), offj:(offj+tmp.shape[1])]=tmp
                if (i!=j) :
                    tmp2=tmp.transpose()
                    v[offj:(offj+tmp2.shape[0]), offi:(offi+tmp2.shape[1])]=tmp2
                offj+=shell_to_basis(self.basis[j].m)
                end = timer()
                time+=end-start
            offi+=shell_to_basis(self.basis[i].m)
        del xyz_all
        del proton_all
        print("V construction from Python takes %f sec." % time) # Time in seconds
        print(v)

    def _get_V_C(self):
        v = np.zeros(shape=(self.nbs,self.nbs));
        time=0.0
        offi=0
        for i, x in enumerate(self.basis):
            offj=0
            for j in range(i+1):
                start = timer()
                tmp = np.zeros(shape=(shell_to_basis(self.basis[i].m),shell_to_basis(self.basis[j].m)))
                for k, xyz3 in enumerate(self.xyz):
                    tmp+=V_C(self.basis[i].c, self.basis[i].p, self.basis[i].n, self.basis[i].e, self.basis[i].m, self.basis[i].xyz,
                             self.basis[j].c, self.basis[j].p, self.basis[j].n, self.basis[j].e, self.basis[j].m, self.basis[j].xyz, xyz3)*self.proton[k]
                    end = timer()
                    time+=end-start
                v[offi:(offi+tmp.shape[0]), offj:(offj+tmp.shape[1])]=tmp
                if (i!=j) :
                    tmp2=tmp.transpose()
                    v[offj:(offj+tmp2.shape[0]), offi:(offi+tmp2.shape[1])]=tmp2
                end = timer()
                time+=end-start

                offj+=shell_to_basis(self.basis[j].m)
            offi+=shell_to_basis(self.basis[i].m)
        print("V construction from C/C++ takes %f sec" % time) # Time in seconds
        print(v)

    def _get_ERI(self):
        eri = np.zeros(shape=(self.nbs,self.nbs,self.nbs,self.nbs));
        time=0.0
        offi=0
        for i, x in enumerate(self.basis):
            offj=0
            for j in range(i+1):
                offk=0
                ij = (i*(i+1)//2 + j)
                for k, y in enumerate(self.basis):
                    offl=0
                    for l in range(k+1):
                        kl = (k*(k+1)//2 + l)
                        if(kl>ij): pass
                        start = timer()
                        tmp1=ERI(self.basis[i].c, self.basis[i].p, self.basis[i].n, self.basis[i].e, self.basis[i].m, self.basis[i].xyz,
                                 self.basis[j].c, self.basis[j].p, self.basis[j].n, self.basis[j].e, self.basis[j].m, self.basis[j].xyz,
                                 self.basis[k].c, self.basis[k].p, self.basis[k].n, self.basis[k].e, self.basis[k].m, self.basis[k].xyz,
                                 self.basis[l].c, self.basis[l].p, self.basis[l].n, self.basis[l].e, self.basis[l].m, self.basis[l].xyz)

                        eri[offi:(offi+tmp1.shape[0]), offj:(offj+tmp1.shape[1]), offk:(offk+tmp1.shape[2]), offl:(offl+tmp1.shape[3])]=tmp1      
                        tmp2=tmp1.transpose((1,0,2,3))
                        eri[offj:(offj+tmp2.shape[0]), offi:(offi+tmp2.shape[1]), offk:(offk+tmp2.shape[2]), offl:(offl+tmp2.shape[3])]=tmp2
                        tmp2=tmp1.transpose((0,1,3,2))
                        eri[offi:(offi+tmp2.shape[0]), offj:(offj+tmp2.shape[1]), offl:(offl+tmp2.shape[2]), offk:(offk+tmp2.shape[3])]=tmp2
                        tmp2=tmp1.transpose((1,0,3,2))
                        eri[offj:(offj+tmp2.shape[0]), offi:(offi+tmp2.shape[1]), offl:(offl+tmp2.shape[2]), offk:(offk+tmp2.shape[3])]=tmp2

                        tmp2=tmp1.transpose((2,3,0,1))
                        eri[offk:(offk+tmp2.shape[0]), offl:(offl+tmp2.shape[1]), offi:(offi+tmp2.shape[2]), offj:(offj+tmp2.shape[3])]=tmp2
                        tmp2=tmp1.transpose((3,2,0,1))
                        eri[offl:(offl+tmp2.shape[0]), offk:(offk+tmp2.shape[1]), offi:(offi+tmp2.shape[2]), offj:(offj+tmp2.shape[3])]=tmp2
                        tmp2=tmp1.transpose((2,3,1,0))
                        eri[offk:(offk+tmp2.shape[0]), offl:(offl+tmp2.shape[1]), offj:(offj+tmp2.shape[2]), offi:(offi+tmp2.shape[3])]=tmp2
                        tmp2=tmp1.transpose((3,2,1,0))
                        eri[offl:(offl+tmp2.shape[0]), offk:(offk+tmp2.shape[1]), offj:(offj+tmp2.shape[2]), offi:(offi+tmp2.shape[3])]=tmp2

                        end = timer()
                        time+=end-start

                        offl+=shell_to_basis(self.basis[l].m)
                    offk+=shell_to_basis(self.basis[k].m)
                offj+=shell_to_basis(self.basis[j].m)
            offi+=shell_to_basis(self.basis[i].m)
        print("ERI construction from Python takes %f sec" % time) # Time in seconds
        print(eri)

    #def _write_basis_f(self,file,i):
    #    file.write("%d  %d\n" % (self.basis[i].c.shape[0], self.basis[i].m.sum()))
    #    for N in range(self.basis[i].c.shape[0]):
    #        file.write("%16.10f %16.10f %16.10f\n" % (self.basis[i].c[N],self.basis[i].n[N],self.basis[i].e[N]))
    #    file.write("%16.10f %16.10f %16.10f\n" % (self.basis[i].xyz[0],self.basis[i].xyz[1],self.basis[i].xyz[2]))

    def _get_ERI_C(self):
        eri = np.zeros(shape=(self.nbs,self.nbs,self.nbs,self.nbs));
        time=0.0
        offi=0
        for i, x in enumerate(self.basis):
            offj=0
            for j in range(i+1):
                offk=0
                ij = (i*(i+1)//2 + j)
                for k, y in enumerate(self.basis):
                    offl=0
                    for l in range(k+1):
                        kl = (k*(k+1)//2 + l)
                        if(kl>ij): pass
                        start = timer()
                        tmp1=ERI_C(self.basis[i].c, self.basis[i].p, self.basis[i].n, self.basis[i].e, self.basis[i].m, self.basis[i].xyz,
                                   self.basis[j].c, self.basis[j].p, self.basis[j].n, self.basis[j].e, self.basis[j].m, self.basis[j].xyz,
                                   self.basis[k].c, self.basis[k].p, self.basis[k].n, self.basis[k].e, self.basis[k].m, self.basis[k].xyz,
                                   self.basis[l].c, self.basis[l].p, self.basis[l].n, self.basis[l].e, self.basis[l].m, self.basis[l].xyz)

                        eri[offi:(offi+tmp1.shape[0]), offj:(offj+tmp1.shape[1]), offk:(offk+tmp1.shape[2]), offl:(offl+tmp1.shape[3])]=tmp1
                        tmp2=tmp1.transpose((1,0,2,3))
                        eri[offj:(offj+tmp2.shape[0]), offi:(offi+tmp2.shape[1]), offk:(offk+tmp2.shape[2]), offl:(offl+tmp2.shape[3])]=tmp2
                        tmp2=tmp1.transpose((0,1,3,2))
                        eri[offi:(offi+tmp2.shape[0]), offj:(offj+tmp2.shape[1]), offl:(offl+tmp2.shape[2]), offk:(offk+tmp2.shape[3])]=tmp2
                        tmp2=tmp1.transpose((1,0,3,2))
                        eri[offj:(offj+tmp2.shape[0]), offi:(offi+tmp2.shape[1]), offl:(offl+tmp2.shape[2]), offk:(offk+tmp2.shape[3])]=tmp2

                        tmp2=tmp1.transpose((2,3,0,1))
                        eri[offk:(offk+tmp2.shape[0]), offl:(offl+tmp2.shape[1]), offi:(offi+tmp2.shape[2]), offj:(offj+tmp2.shape[3])]=tmp2
                        tmp2=tmp1.transpose((3,2,0,1))
                        eri[offl:(offl+tmp2.shape[0]), offk:(offk+tmp2.shape[1]), offi:(offi+tmp2.shape[2]), offj:(offj+tmp2.shape[3])]=tmp2
                        tmp2=tmp1.transpose((2,3,1,0))
                        eri[offk:(offk+tmp2.shape[0]), offl:(offl+tmp2.shape[1]), offj:(offj+tmp2.shape[2]), offi:(offi+tmp2.shape[3])]=tmp2
                        tmp2=tmp1.transpose((3,2,1,0))
                        eri[offl:(offl+tmp2.shape[0]), offk:(offk+tmp2.shape[1]), offj:(offj+tmp2.shape[2]), offi:(offi+tmp2.shape[3])]=tmp2

                        end = timer()
                        time+=end-start

                        offl+=shell_to_basis(self.basis[l].m)
                    offk+=shell_to_basis(self.basis[k].m)
                offj+=shell_to_basis(self.basis[j].m)
            offi+=shell_to_basis(self.basis[i].m)
        print("ERI construction from C/C++ takes %f sec" % time) # Time in seconds
        print(eri)

    def _get_ERI_L(self):
        eri = np.zeros(shape=(self.nbs,self.nbs,self.nbs,self.nbs));
        time=0.0
        offi=0
        for i, x in enumerate(self.basis):
            offj=0
            for j in range(i+1):
                offk=0
                ij = (i*(i+1)//2 + j)
                for k, y in enumerate(self.basis):
                    offl=0
                    for l in range(k+1):
                        kl = (k*(k+1)//2 + l)
                        if(kl>ij): pass
                        start = timer()
                        tmp1=ERI_L(self.basis[i].c, self.basis[i].p, self.basis[i].n, self.basis[i].e, self.basis[i].m, self.basis[i].xyz,
                                   self.basis[j].c, self.basis[j].p, self.basis[j].n, self.basis[j].e, self.basis[j].m, self.basis[j].xyz,
                                   self.basis[k].c, self.basis[k].p, self.basis[k].n, self.basis[k].e, self.basis[k].m, self.basis[k].xyz,
                                   self.basis[l].c, self.basis[l].p, self.basis[l].n, self.basis[l].e, self.basis[l].m, self.basis[l].xyz)

                        eri[offi:(offi+tmp1.shape[0]), offj:(offj+tmp1.shape[1]), offk:(offk+tmp1.shape[2]), offl:(offl+tmp1.shape[3])]=tmp1
                        tmp2=tmp1.transpose((1,0,2,3))
                        eri[offj:(offj+tmp2.shape[0]), offi:(offi+tmp2.shape[1]), offk:(offk+tmp2.shape[2]), offl:(offl+tmp2.shape[3])]=tmp2
                        tmp2=tmp1.transpose((0,1,3,2))
                        eri[offi:(offi+tmp2.shape[0]), offj:(offj+tmp2.shape[1]), offl:(offl+tmp2.shape[2]), offk:(offk+tmp2.shape[3])]=tmp2
                        tmp2=tmp1.transpose((1,0,3,2))
                        eri[offj:(offj+tmp2.shape[0]), offi:(offi+tmp2.shape[1]), offl:(offl+tmp2.shape[2]), offk:(offk+tmp2.shape[3])]=tmp2

                        tmp2=tmp1.transpose((2,3,0,1))
                        eri[offk:(offk+tmp2.shape[0]), offl:(offl+tmp2.shape[1]), offi:(offi+tmp2.shape[2]), offj:(offj+tmp2.shape[3])]=tmp2
                        tmp2=tmp1.transpose((3,2,0,1))
                        eri[offl:(offl+tmp2.shape[0]), offk:(offk+tmp2.shape[1]), offi:(offi+tmp2.shape[2]), offj:(offj+tmp2.shape[3])]=tmp2
                        tmp2=tmp1.transpose((2,3,1,0))
                        eri[offk:(offk+tmp2.shape[0]), offl:(offl+tmp2.shape[1]), offj:(offj+tmp2.shape[2]), offi:(offi+tmp2.shape[3])]=tmp2
                        tmp2=tmp1.transpose((3,2,1,0))
                        eri[offl:(offl+tmp2.shape[0]), offk:(offk+tmp2.shape[1]), offj:(offj+tmp2.shape[2]), offi:(offi+tmp2.shape[3])]=tmp2

                        end = timer()
                        time+=end-start

                        offl+=shell_to_basis(self.basis[l].m)
                    offk+=shell_to_basis(self.basis[k].m)
                offj+=shell_to_basis(self.basis[j].m)
            offi+=shell_to_basis(self.basis[i].m)
        print("ERI construction from Libint takes %f sec" % time) # Time in seconds
        print(eri)
 
class Basis():
    """
    A class handling basis set objects

    """
    def __init__(self, coef=None, exp=None, mom=None, xyz=None):
        self.xyz    = xyz
        self.m      = mom
        self.e      = exp
        self.c      = coef
        self.p      = None
        self.n      = None
        self._normalize()

    def _print(self):
        print(self.c, self.e, self.m, self.n, self.xyz)

    def _normalize(self):
        ''' Routine to normalize both prim Gaussian and 
            contracted basis functions.
            The prim Gaussian is normalized to x, xx, xxx .....
            The scale factor is multiplied to n for different
            angular momentum.
            Such complicated arrangement is due to the interface to Libint.
        '''
        L  = self.m
        mx = self.m
        my = 0
        mz = 0
        self.p  = np.sqrt(np.power(2,2*(L)+1.5)*
                  np.power(self.e,L+1.5)/
                  fact2(2*mx-1)/fact2(2*my-1)/
                  fact2(2*mz-1)/np.power(np.pi,1.5))

        self.n = np.zeros(shape=(shell_to_basis(self.m)))
        fac_xyz0= 1.0*fact2(2*mx-1)*fact2(2*my-1)*fact2(2*mz-1)

        offi=0
        for k1 in range(self.m+1):
            mx=self.m-k1
            for mz in range(k1+1):
                my = k1 - mz;

                # n is a list of length equal to number primitives
                # normalize primitives first, but we do not save it.
                fac_xyz= 1.0*fact2(2*mx-1)*fact2(2*my-1)*fact2(2*mz-1)
                n = np.sqrt(np.power(2,2*(L)+1.5)*
                            np.power(self.e,L+1.5)/
                            fac_xyz/np.power(np.pi,1.5))

                # now normalize the contracted basis functions
                prefactor = np.power(np.pi,1.5)*fac_xyz/np.power(2.0,L)

                N=0.0
                num_exps = len(self.e)
                for ia in range(self.c.shape[0]):
                    for ib in range(self.c.shape[0]):
                        N += n[ia]*n[ib]*self.c[ia]*self.c[ib]/\
                             np.power(self.e[ia] + self.e[ib],L+1.5)

                # Here N is scaled with respect to the 
                # first element belonging to this am for each elememt
                N *= prefactor*fac_xyz/fac_xyz0
                N = np.power(N,-0.5)
                self.n[offi]=N
                offi+=1

if __name__ == '__main__':
    start = timer()
    h2o = Geometry('h2o', 
                    coor1='''O  0. 0. 0.
                            O  1. 0. 0.
                            O  2. 0. 0.
                            O  3. 0. 0.
                            O  4. 0. 0.
                            O  5. 0. 0.
                            O  6. 0. 0.
                            O  7. 0. 0.
                            O  8. 0. 0.
                            O  9. 0. 0.
                            O  10. 0. 0.
                            O  1. 0. 0.
                            O  2. 0. 0.
                            O  3. 0. 0.
                            O  4. 0. 0.
                            O  5. 0. 0.
                            O  6. 0. 0.
                            O  7. 0. 0.
                            O  8. 0. 0.
                            O  9. 0. 0.
                            O  10. 0. 0.
                            O  1. 0. 0.
                            O  2. 0. 0.
                            O  3. 0. 0.
                            O  4. 0. 0.
                            O  5. 0. 0.
                            O  6. 0. 0.
                            O  7. 0. 0.
                            O  8. 0. 0.
                            O  9. 0. 0.
                            O  10. 0. 0.
                            O  1. 0. 0.
                            O  2. 0. 0.
                            O  3. 0. 0.
                            O  4. 0. 0.
                            O  5. 0. 0.
                            O  6. 0. 0.
                            O  7. 0. 0.
                            O  8. 0. 0.
                            O  9. 0. 0.
                            O  10. 0. 0.
                            O  10. 0. 0.
                            O  1. 0. 0.
                            O  2. 0. 0.
                            O  3. 0. 0.
                            O  4. 0. 0.
                            O  5. 0. 0.
                            O  6. 0. 0.
                            O  7. 0. 0.
                            O  8. 0. 0.
                            O  9. 0. 0.
                            O  10. 0. 0.
                            O  1. 0. 0.
                            O  2. 0. 0.
                            O  3. 0. 0.
                            O  4. 0. 0.
                            O  5. 0. 0.
                            O  6. 0. 0.
                            O  7. 0. 0.
                            O  8. 0. 0.
                            O  9. 0. 0.
                            O  10. 0. 0.
                            O  11. 0. 0.''',
                   coor='''H  0. 1.2 0.
                           H  1.1 0. 0.
                           H  0. 0. 0.
                           H  2.1 1.2 0.
                           ''',
#coor='''C       -0.558305038     -0.000000250     -2.716159087
#H       -1.578997038     -0.008907250     -2.317139087
#H       -0.040245038     -0.879491250     -2.317139087
#H       -0.055673038      0.888398750     -2.317139087
#C       -0.558305038     -0.000000250     -4.256159087
#H        0.462386962     -0.008907250     -4.655179087
#H       -1.076365038     -0.879491250     -4.655179087
#H       -1.060937038      0.888398750     -4.655179087''',
                   charge=3,
                   multi=4,
                   basisname='cc-pvdz')
    print(h2o.charge)
    print(h2o.multi)
    end = timer()
    print(end - start) # Time in seconds


