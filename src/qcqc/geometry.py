#!/usr/bin/env python

import basis_set_exchange as bse
from  basis_set_exchange import lut
import parameter
import qcqc

class Geometry:
    """
    A class handling geometry objects
 
    """
    def __init__(self, name=None, atoms=[], **kwargs):
        self.name   = name
        self.elem   = []
        self.proton = []
        self.x      = []
        self.y      = []
        self.z      = []
        self.basis  = []
        self.charge = int(kwargs.pop('charge',parameter._DEFAULT_CHARGE))
        self.multi  = int(kwargs.pop('multi', parameter._DEFAULT_MULTI))
        self.bsname = kwargs.pop('basisname', parameter._DEFAULT_BASIS)
        if atoms: self._read_atoms(atoms)
        return

    def _read_atoms(self, atoms):

        for id, atom in enumerate(atoms):
            residue=id%4
            if (residue==0):
                self.elem.append(atom)
            elif(residue==1):
                self.x.append(float(atom))
            elif(residue==2):
                self.y.append(float(atom))
            elif(residue==3):
                self.z.append(float(atom))

        print("Number  Elements          X                Y               Z")
        for id, atom in enumerate(self.elem):
            print("%6d  %5s   %16.10f %16.10f %16.10f" % (id,self.elem[id],self.x[id],self.y[id],self.z[id]))
        #Get basis information for each shell
        Basis(self)
    

class Basis(Geometry):
    """
    A class handling basis set objects

    """
    def __init__(self,Geometry):
        self.x      = []
        self.y      = []
        self.z      = []
        self.m      = []
        self.e      = []
        self.c      = []
        self.bsname = Geometry.bsname 
        self._get_basis(Geometry)
        self._print()
        return


    def _get_basis(self,Geometry):

        for id, elem in enumerate(Geometry.elem):
            sym, number, name = lut.element_data_from_sym(elem)
            bs_str = bse.get_basis(self.bsname, elements=sym, header=False)
            Geometry.basis.append(bs_str)
        for id, atom in enumerate(Geometry.elem):
            #print("%6d %s %16.10f %16.10f %16.10f" % (id,atom,Geometry.x[id],Geometry.y[id],Geometry.z[id]))

            for k, el in Geometry.basis[id]['basis_set_elements'].items():
                if not 'element_electron_shells' in el:
                    continue

                for sh in el['element_electron_shells']:
                    exponents = sh['shell_exponents']
                    # transpose of the coefficient matrix
                    coeff_t = sh['shell_coefficients']
                    am = sh['shell_angular_momentum']
                    for counter, value in enumerate(am):
                        #assign XYZ for each shell
                        self.x.append(Geometry.x[id])
                        self.y.append(Geometry.y[id])
                        self.z.append(Geometry.z[id])
                        #assign angular_momentum for each shell
                        self.m.append(value)
                        #assign exponents for each shell
                        self.e.append(list(map(float, exponents)))
                        #assign coefficients for each shell
                        self.c.append(list(map(float, coeff_t[counter])))

    def _print(self):
        print('\n')
        for shell, x in enumerate(self.x):
            print("%6d %6d %16.10f %16.10f %16.10f" % (shell,self.m[shell],self.x[shell],self.y[shell],self.z[shell]))
            #for ngto in range(len(self.e[shell])):
            for ngto, value in enumerate(self.e[shell]):
                print("%6d  %16.10f %16.10f" % (ngto, self.c[shell][ngto],self.e[shell][ngto]))
            print(qcqc.subtract(333, 111))

if __name__ == '__main__':
    h2o = Geometry('h2o',
                   ['O',0.,0.,0.,
                    'H',1.,0.,0.,
                    'H',0.,1.,0.],
                   charge=3,
                   multi=4,
                   basisname='sto-3g')
    #Basis=Basis(h2o)
    print(h2o.charge)
    print(h2o.multi)


