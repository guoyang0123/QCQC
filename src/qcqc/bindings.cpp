#include <pybind11/pybind11.h>
#include "math.hpp"

namespace py = pybind11;

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
  
  m.def("eri", &contr_coulomb, "Compute two-electron integrals")
}
