#!/usr/bin/env python

import basis_set_exchange as bse
from  basis_set_exchange import lut
import parameter

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
        self._get_basis()
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

    def _get_basis(self):

        for id, elem in enumerate(self.elem):
            sym, number, name = lut.element_data_from_sym(elem)
            self.proton.append(int(number))
            bs_str = bse.get_basis(self.bsname, elements=sym, header=False)
            self.basis.append(bs_str)
        for id, atom in enumerate(self.elem):
            print("%6d  %5d   %16.10f %16.10f %16.10f" % (id,self.proton[id],self.x[id],self.y[id],self.z[id]))

            for k, el in self.basis[id]['basis_set_elements'].items():
                if not 'element_electron_shells' in el:
                    continue

                for sh in el['element_electron_shells']:
                    exponents = sh['shell_exponents']
                    # transpose of the coefficient matrix
                    coeff_t = sh['shell_coefficients']
                    am = sh['shell_angular_momentum']
                    for counter, value in enumerate(am):
                        print(value, coeff_t[counter], exponents)

if __name__ == '__main__':
    h2o = Geometry('h2o',
                   ['O',0.,0.,0.,
                    'H',1.,0.,0.,
                    'H',0.,1.,0.],
                   charge=3,
                   multi=4,
                   basisname='sto-3g')
    print(h2o.charge)
    print(h2o.multi)


