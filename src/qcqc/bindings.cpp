#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "math.hpp"
#include "cints.hpp"
#include "libint.h"
#include <iostream>

namespace py = pybind11;

void py_compute_eri(int np1, py::array_t<double> pc1, py::array_t<double> pp1, py::array_t<double> pn1, py::array_t<double> pe1, int am1, double x1, double y1, double z1,
                    int np2, py::array_t<double> pc2, py::array_t<double> pp2, py::array_t<double> pn2, py::array_t<double> pe2, int am2, double x2, double y2, double z2,
                    int np3, py::array_t<double> pc3, py::array_t<double> pp3, py::array_t<double> pn3, py::array_t<double> pe3, int am3, double x3, double y3, double z3,
                    int np4, py::array_t<double> pc4, py::array_t<double> pp4, py::array_t<double> pn4, py::array_t<double> pe4, int am4, double x4, double y4, double z4,
                    py::array_t<double> peri){

  int p, q, r, s;

  int nb1, nb2, nb3, nb4; // Number of Cartesian basis function

  nb1 = (am1 + 1)*(am1 + 2)/2;
  nb2 = (am2 + 1)*(am2 + 2)/2;
  nb3 = (am3 + 1)*(am3 + 2)/2;
  nb4 = (am4 + 1)*(am4 + 2)/2;

  //Interface to access numpy array
  auto buf_c1 = pc1.request(), buf_c2 = pc2.request(),buf_c3 = pc3.request(), buf_c4 = pc4.request();
  auto buf_p1 = pp1.request(), buf_p2 = pp2.request(),buf_p3 = pp3.request(), buf_p4 = pp4.request();
  auto buf_n1 = pn1.request(), buf_n2 = pn2.request(),buf_n3 = pn3.request(), buf_n4 = pn4.request();
  auto buf_e1 = pe1.request(), buf_e2 = pe2.request(),buf_e3 = pe3.request(), buf_e4 = pe4.request();
  auto buf_eri= peri.request();

  if (buf_c1.shape[0]!=np1 || buf_c2.shape[0]!=np2 || buf_c3.shape[0]!=np3 || buf_c4.shape[0]!=np4) 
      throw std::runtime_error("Input c dim must match nprim");
  if (buf_p1.shape[0]!=np1 || buf_p2.shape[0]!=np2 || buf_p3.shape[0]!=np3 || buf_p4.shape[0]!=np4) 
      throw std::runtime_error("Input p dim must match nprim");
  if (buf_n1.shape[0]!=nb1 || buf_n2.shape[0]!=nb2 || buf_n3.shape[0]!=nb3 || buf_n4.shape[0]!=nb4)
      throw std::runtime_error("Input n dim must match nb");
  if (buf_e1.shape[0]!=np1 || buf_e2.shape[0]!=np2 || buf_e3.shape[0]!=np3 || buf_e4.shape[0]!=np4) 
      throw std::runtime_error("Input e dim must match nprim");
  if (buf_eri.shape[0]!=nb1*nb2*nb3*nb4) 
      throw std::runtime_error("Input eri dim must match nb1*nb2*nb3*nb4");

  // Define pointers to numpy arrays
  double *c1 = (double *) buf_c1.ptr,
         *c2 = (double *) buf_c2.ptr,
         *c3 = (double *) buf_c3.ptr,
         *c4 = (double *) buf_c4.ptr;

  double *p1 = (double *) buf_p1.ptr,
         *p2 = (double *) buf_p2.ptr,
         *p3 = (double *) buf_p3.ptr,
         *p4 = (double *) buf_p4.ptr;

  double *n1 = (double *) buf_n1.ptr,
         *n2 = (double *) buf_n2.ptr,
         *n3 = (double *) buf_n3.ptr,
         *n4 = (double *) buf_n4.ptr;

  double *e1 = (double *) buf_e1.ptr,
         *e2 = (double *) buf_e2.ptr,
         *e3 = (double *) buf_e3.ptr,
         *e4 = (double *) buf_e4.ptr;

  double *eri= (double *) buf_eri.ptr;

  //std::cout << " before swap"<< z4 << std::endl;
  // Libint only provides code to computes a permutation-symmetry-unique subset of integrals
  // the default convention is to compute integrals (1 2|3 4) with am1 >= am2, am3 >= am4, and am3+am4 >= am1+am2
  // We have to swap original (1 2|3 4) to the correct order
  bool swap12=0, swap34=0, s13s24=0;

  if (am1 < am2){

    std::swap(c1, c2);
    std::swap(p1, p2);
    std::swap(n1, n2);
    std::swap(e1, e2);
    std::swap(np1, np2);
    std::swap(am1, am2);
    std::swap(x1, x2);
    std::swap(y1, y2);
    std::swap(z1, z2);
    swap12 = 1;
  }
  if (am3 < am4){

    std::swap(c3, c4);
    std::swap(p3, p4);
    std::swap(n3, n4);
    std::swap(e3, e4);
    std::swap(np3, np4);
    std::swap(am3, am4);
    std::swap(x3, x4);
    std::swap(y3, y4);
    std::swap(z3, z4);
    swap34 = 1;
  }
  if ((am1+am2) > (am3+am4)){

    std::swap(c1, c3);
    std::swap(p1, p3);
    std::swap(n1, n3);
    std::swap(e1, e3);
    std::swap(np1, np3);
    std::swap(am1, am3);
    std::swap(x1, x3);
    std::swap(y1, y3);
    std::swap(z1, z3);

    std::swap(c2, c4);
    std::swap(p2, p4);
    std::swap(n2, n4);
    std::swap(e2, e4);
    std::swap(np2, np4);
    std::swap(am2, am4);
    std::swap(x2, x4);
    std::swap(y2, y4);
    std::swap(z2, z4);

    s13s24 = 1;
  }
  //std::cout << " swap12 ="<< swap12 <<", swap34 ="<< swap34  <<", s13s24 ="<< s13s24 << std::endl;

  //std::cout << " before compute_eri"<< z4 << std::endl;
  double *eri_tmp;
  eri_tmp=init_array(nb1*nb2*nb3*nb4);
  compute_eri(np1, c1, p1, e1, am1, x1, y1, z1,
              np2, c2, p2, e2, am2, x2, y2, z2,
              np3, c3, p3, e3, am3, x3, y3, z3,
              np4, c4, p4, e4, am4, x4, y4, z4,
              eri_tmp);

  //int N=nb1*nb2*nb3*nb4;
  //for (int i=0; i<N; i++)
  //{
  //  std::cout << eri_tmp[i]<< std::endl;
  //}

  //std::cout << " before compute_pqrs"<< z4 << std::endl;
  nb1 = (am1 + 1)*(am1 + 2)/2;
  nb2 = (am2 + 1)*(am2 + 2)/2;
  nb3 = (am3 + 1)*(am3 + 2)/2;
  nb4 = (am4 + 1)*(am4 + 2)/2;
  for (p = 0; p < nb1; p++){
    for (q = 0; q < nb2; q++){
      for (r = 0; r < nb3 ; r++){
        for (s = 0; s < nb4 ; s++){
           int old_pqrs = compute_pqrs(p, q, r, s, nb1, nb2, nb3, nb4,
                                       swap12, swap34, s13s24);
           int pqrs     = ((p * nb2 + q)* nb3 + r) * nb4 + s;
           eri[old_pqrs] = eri_tmp[pqrs]*n1[p]*n2[q]*n3[r]*n4[s];
           //eri[old_pqrs] = eri_tmp[pqrs];
        }
      }
    }
  }

  if (s13s24){

    std::swap(c1, c3);
    std::swap(p1, p3);
    std::swap(n1, n3);
    std::swap(e1, e3);
    std::swap(np1, np3);
    std::swap(am1, am3);

    std::swap(c2, c4);
    std::swap(p2, p4);
    std::swap(n2, n4);
    std::swap(e2, e4);
    std::swap(np2, np4);
    std::swap(am2, am4);
  }

  if (swap34){

    std::swap(c3, c4);
    std::swap(p3, p4);
    std::swap(n3, n4);
    std::swap(e3, e4);
    std::swap(np3, np4);
    std::swap(am3, am4);
  }

  if (swap12){

    std::swap(c1, c2);
    std::swap(p1, p2);
    std::swap(n1, n2);
    std::swap(e1, e2);
    std::swap(np1, np2);
    std::swap(am1, am2);
  }

  //for (int i=0; i<N; i++)
  //{
  //  std::cout << eri[i]<< std::endl;
  //}

  free_array(eri_tmp);
}


PYBIND11_MODULE(qcqc, m) {

  ///////////////////////////////////////////////////////////
  // This two examples are left here intentionally,
  // as the starting point of our qcqc project.
  // "PYBIND11_MODULE" is used, according to 
  // latest pybind11 manual, which is different 
  // from 'https://github.com/benjaminjack/python_cpp_example'.
  // The pybind11 in qcqc/lib is updated to date as well. 
  m.def("add", &add, "Add two numbers");
  m.def("subtract", &subtract, "Subtract two numbers");
  /////////////////////////////////////////////////////////// 
  m.def("overlap", &overlap, "Compute overlap integrals over two Gaussian");
  m.def("kinetic", &kinetic, "Compute kinetic integrals over two Gaussian");
  m.def("nuclear", &nuclear_attraction, "Compute nuclear attraction integrals over two Gaussian");
  m.def("coulomb_repulsion", &coulomb_repulsion, "Compute two electron integrals over four Gaussian");
  m.def("compute_eri", &py_compute_eri, "Compute two electron integrals by libint");
}
