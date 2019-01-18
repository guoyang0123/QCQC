#include <fstream>
#include <sstream>
#include <string>
#include <iostream> 
#include <new>

#include "libint.h"
#include <cmath>
#include <algorithm>
#define MAXFAC 100
#define EPS 1.0E-17
static double *df;

double* init_array(int size) {
  double* result = new double[size];
  for (int i = 0; i < size; i++)
    result[i] = 0.0;
  return result;
}

void free_array(double* array) {
  delete[] array;
}

/*!
 calc_f()

 This function computes infamous integral Fn(t). For its definition
 see Obara and Saika paper, or Shavitt's chapter in the
 Methods in Computational Physics book (see reference below).
 This piece of code is from Dr. Justin Fermann's program CINTS

 \ingroup (QT)
 */
void calc_f(double *F, int n, double t) {
  int i, m, k;
  int m2;
  double t2;
  double num;
  double sum;
  double term1, term2;
  static double K = 1.0 / M_2_SQRTPI;
  double et;

  if (df == NULL) {
    df = init_array(2 * MAXFAC);
    df[0] = 1.0;
    df[1] = 1.0;
    df[2] = 1.0;
    for (i = 3; i < MAXFAC * 2; i++) {
      df[i] = (i - 1) * df[i - 2];
    }
  }

  if (t > 20.0) { /* For big t's do upward recursion */
    t2 = 2 * t;
    et = exp(-t);
    t = sqrt(t);
    F[0] = K * erf(t) / t;
    for (m = 0; m <= n - 1; m++) {
      F[m + 1] = ((2 * m + 1) * F[m] - et) / (t2);
    }
  } else { /* For smaller t's compute F with highest n using
   asymptotic series (see I. Shavitt in
   Methods in Computational Physics, ed. B. Alder eta l,
   vol 2, 1963, page 8) */
    et = exp(-t);
    t2 = 2 * t;
    m2 = 2 * n;
    num = df[m2];
    i = 0;
    sum = 1.0 / (m2 + 1);
    do {
      i++;
      num = num * t2;
      term1 = num / df[m2 + 2 * i + 2];
      sum += term1;
    } while (fabs(term1) > EPS && i < MAXFAC);
    F[n] = sum * et;
    for (m = n - 1; m >= 0; m--) { /* And then do downward recursion */
      F[m] = (t2 * F[m + 1] + et) / (2 * m + 1);
    }
  }
}

double norm_const(int l1, int m1, int n1, double alpha1) {
  return pow(2 * alpha1 / M_PI, 0.75) * pow(4 * alpha1, 0.5 * (l1 + m1 + n1))
      / sqrt(df[2 * l1] * df[2 * m1] * df[2 * n1]);
}

void prep_libint(prim_data &erieval,  
                 double c1, double p1, int m1, double e1, double *A, 
                 double c2, double p2, int m2, double e2, double *B,
                 double c3, double p3, int m3, double e3, double *C,
                 double c4, double p4, int m4, double e4, double *D) {

  const int mmax = m1 + m2 + m3 + m4;
  double P[3], Q[3], W[3];

  double z  = e1 + e2;
  double n  = e3 + e4;
  double rho= z*n/(z+n);

  for (int i = 0; i < 3 ; i++)
  {
      P[i] = (A[i]* e1 + B[i] * e2) / z;
      Q[i] = (C[i]* e3 + D[i] * e4) / n;
      W[i] = (P[i]* z  + Q[i] * n ) / (z+n);
  }

  /* Information in prim_data of libint.h
  typedef struct pdata{
  REALTYPE F[25];
  REALTYPE U[6][3];
  REALTYPE twozeta_a;
  REALTYPE twozeta_b;
  REALTYPE twozeta_c;
  REALTYPE twozeta_d;
  REALTYPE oo2z;
  REALTYPE oo2n;
  REALTYPE oo2zn;
  REALTYPE poz;
  REALTYPE pon;
  REALTYPE oo2p;
  REALTYPE ss_r12_ss;
  } prim_data;i  */

  double AB2 = (A[0] - B[0])*(A[0] - B[0])+ 
               (A[1] - B[1])*(A[1] - B[1])+
               (A[2] - B[2])*(A[2] - B[2]);
  double CD2 = (C[0] - D[0])*(C[0] - D[0])+ 
               (C[1] - D[1])*(C[1] - D[1])+
               (C[2] - D[2])*(C[2] - D[2]);
  double PQ2 = (P[0] - Q[0])*(P[0] - Q[0])+
               (P[1] - Q[1])*(P[1] - Q[1])+
               (P[2] - Q[2])*(P[2] - Q[2]); 

  double K1 = exp(-e1 * e2 * AB2 / z)*pow(M_PI/z, 1.5);
  double K2 = exp(-e3 * e4 * CD2 / n)*pow(M_PI/n, 1.5);
  double pfac = 2.0 * K1 * K2 * sqrt(rho/M_PI);

  //p is the normalization factor 
  //of prim Gaussian in the first
  //element within this am
  pfac *= c1*c2*c3*c4*p1*p2*p3*p4;
  double* F = init_array(mmax + 1);

  // asign F
  calc_f(F, mmax, PQ2*rho);
  for (int i = 0; i <= mmax; i++){
      erieval.F[i] = F[i] * pfac;
  }

  // asign U 
  for (int i = 0; i < 3; i++) {
    erieval.U[0][i] = P[i]-A[i];
    erieval.U[1][i] = P[i]-B[i];
    erieval.U[2][i] = Q[i]-C[i];
    erieval.U[3][i] = Q[i]-D[i];
    erieval.U[4][i] = W[i]-P[i];
    erieval.U[5][i] = W[i]-Q[i];
  }

  erieval.twozeta_a = 2.0 * e1;
  erieval.twozeta_b = 2.0 * e2;
  erieval.twozeta_c = 2.0 * e3;
  erieval.twozeta_d = 2.0 * e4;
  erieval.oo2z      = 0.5 / z;
  erieval.oo2n      = 0.5 / n;
  erieval.oo2zn     = 1.0 / (2*(z+n));
  erieval.poz       = rho / z;
  erieval.pon       = rho / n;
  erieval.oo2p      = 1.0 / (2 * rho);
}


void compute_eri(int nprim1, double *c1, double *n1, double *e1, int am1, double x1, double y1, double z1,
                 int nprim2, double *c2, double *n2, double *e2, int am2, double x2, double y2, double z2,
                 int nprim3, double *c3, double *n3, double *e3, int am3, double x3, double y3, double z3,
                 int nprim4, double *c4, double *n4, double *e4, int am4, double x4, double y4, double z4, 
                 double *eri){

  int p, q, r, s;

  int nb1, nb2, nb3, nb4; // Number of Cartesian basis function

  nb1 = (am1 + 1)*(am1 + 2)/2;
  nb2 = (am2 + 1)*(am2 + 2)/2;
  nb3 = (am3 + 1)*(am3 + 2)/2;
  nb4 = (am4 + 1)*(am4 + 2)/2;
  
  // Initialize libint static data
  init_libint_base();

  // 1. Maximum angular momentum
  int max_am    = std::max(std::max(am1, am2), std::max(am3, am4));
  // 2. Maximum number of primitive combinations
  int max_nprim = nprim1 * nprim2 * nprim3 * nprim4;

  Libint_t inteval;
  double LL = libint_storage_required(max_am,  max_nprim);
  init_libint(&inteval, max_am , max_nprim);

  int N=0;
  double A[3], B[3], C[3], D[3];
  A[0]=x1; A[1]=y1; A[2]=z1;
  B[0]=x2; B[1]=y2; B[2]=z2;
  C[0]=x3; C[1]=y3; C[2]=z3;
  D[0]=x4; D[1]=y4; D[2]=z4;

  inteval.AB[0] = A[0] - B[0];
  inteval.AB[1] = A[1] - B[1];
  inteval.AB[2] = A[2] - B[2];
  inteval.CD[0] = C[0] - D[0];
  inteval.CD[1] = C[1] - D[1];
  inteval.CD[2] = C[2] - D[2];

  for (p = 0; p < nprim1; p++){
    for (q = 0; q < nprim2; q++){
      for (r = 0; r < nprim3; r++){
        for (s = 0; s < nprim4; s++){
          prep_libint(inteval.PrimQuartet[N],
                      c1[p], n1[p], am1, e1[p], A,
                      c2[q], n2[q], am2, e2[q], B,
                      c3[r], n3[r], am3, e3[r], C,
                      c4[s], n4[s], am4, e4[s], D);
          N=N+1;
        }
      }
    }
  }

  // Invoking Libint
  double *integral;
  if (am1 == 0 && am2 == 0 && am3 == 0 && am4 == 0){
    // Computing (ss|ss)
    double ssss = 0.0;
    for (p = 0; p < max_nprim ; p++){
      ssss += inteval.PrimQuartet[p].F[0];
    }
    eri[0] = ssss;
  }
  else{
    integral = build_eri[am1][am2][am3][am4](&inteval, max_nprim);
  
    for (p = 0; p < nb1; p++){
      for (q = 0; q < nb2; q++){
        for (r = 0; r < nb3; r++){
          for (s = 0; s < nb4; s++){
            int pqrs = ((p * nb2 + q) * nb3 + r) * nb4 + s;
            eri[pqrs] = integral[pqrs];
          }
        }
      }
    }
  }
  //N=nb1*nb2*nb3*nb4;
  //for (int i=0; i<N; i++)
  //{
  //  std::cout << eri[i]<< std::endl;
  //}
  free_libint(&inteval);
}

int compute_pqrs(int p, int q, int r, int s, int n1, int n2, int n3, int n4,
                 bool swap12, bool swap34, bool s13s24){

  if (s13s24){

    std::swap(p, r);
    std::swap(n1,n3);
    std::swap(q, s);
    std::swap(n2,n4);
  }
  if (swap34){
    std::swap(r, s);
    std::swap(n3,n4);
  }
  if (swap12){
    std::swap(p, q);
    std::swap(n1,n2);
  }
  return ((p*n2+q)*n3+r)*n4+s;
}


//int main() {
//
//  std::ifstream infile("basis.txt");
//  std::string line;
//
//  int p1, p2, p3, p4;
//  int m1, m2, m3, m4;
//  double *c1, *c2, *c3, *c4;
//  double *n1, *n2, *n3, *n4;
//  double *e1, *e2, *e3, *e4;
//  double  x1,  x2,  x3,  x4;
//  double  y1,  y2,  y3,  y4;
//  double  z1,  z2,  z3,  z4;
//
//  std::cout << "hello world!" << std::endl;
//
//  //first basis
//  std::getline(infile, line);
//  //std::cout << line, '\n';
//  std::istringstream iss(line);
//
//  iss >> p1 >> m1;
//  std::cout << p1 <<", am="<< m1 << std::endl;
//
//  std::stringstream ss;
//
//  c1 = new double[p1];
//  n1 = new double[p1];
//  e1 = new double[p1];
//  for (int i=0; i<p1; i++) {
//     std::getline(infile, line);
//     ss.clear();
//     ss.str(line);
//     ss >> c1[i] >> n1[i] >> e1[i];
//     std::cout << "c ="<< c1[i] <<", n ="<< n1[i]  <<", e ="<< e1[i] << std::endl;
//  }
//  std::getline(infile, line);
//  iss.clear();
//  iss.str(line);
//  iss >> x1 >> y1 >> z1;
//  std::cout << "x1 ="<< x1 <<", y1 ="<< y1  <<", z1 ="<< z1 << std::endl;
//
//  //second basis
//  std::getline(infile, line);
//  iss.clear();
//  iss.str(line);
//  iss >> p2 >> m2;
//  c2 = new double[p2];
//  n2 = new double[p2];
//  e2 = new double[p2];
//  for (int i=0; i<p2; i++) {
//     std::getline(infile, line);
//     ss.clear();
//     ss.str(line);
//     ss >> c2[i] >> n2[i] >> e2[i];
//     std::cout << "c ="<< c2[i] <<", n ="<< n2[i]  <<", e ="<< e2[i] << std::endl;
//  }
//  std::getline(infile, line);
//  ss.clear();
//  ss.str(line);
//  ss >> x2 >> y2 >> z2;
//  std::cout << "x2 ="<< x2 <<", y2 ="<< y2  <<", z2 ="<< z2 << std::endl;
//
//  //third basis
//  std::getline(infile, line);
//  iss.clear();
//  iss.str(line);
//  iss >> p3 >> m3;
//  c3 = new double[p3];
//  n3 = new double[p3];
//  e3 = new double[p3];
//  for (int i=0; i<p3; i++) {
//     std::getline(infile, line);
//     ss.clear();
//     ss.str(line);
//     ss >> c3[i] >> n3[i] >> e3[i];
//     std::cout << "c ="<< c3[i] <<", n ="<< n3[i]  <<", e ="<< e3[i] << std::endl;
//  }
//  std::getline(infile, line);
//  ss.clear();
//  ss.str(line);
//  ss >> x3 >> y3 >> z3;
//  std::cout << "x3 ="<< x3 <<", y3 ="<< y3  <<", z3 ="<< z3 << std::endl;
//
//  //fourth basis
//  std::getline(infile, line);
//  iss.clear();
//  iss.str(line);
//  iss >> p4 >> m4;
//  c4 = new double[p4];
//  n4 = new double[p4];
//  e4 = new double[p4];
//  for (int i=0; i<p4; i++) {
//     std::getline(infile, line);
//     ss.clear();
//     ss.str(line);
//     ss >> c4[i] >> n4[i] >> e4[i];
//     std::cout << "c ="<< c4[i] <<", n ="<< n4[i]  <<", e ="<< e4[i] << std::endl;
//  }
//  std::getline(infile, line);
//  ss.clear();
//  ss.str(line);
//  ss >> x4 >> y4 >> z4;
//  std::cout << "x4 ="<< x4 <<", y4 ="<< y4  <<", z4 ="<< z4 << std::endl;
//
//  int nb1, nb2, nb3, nb4; // Number of Cartesian basis function
//
//  nb1 = (m1 + 1)*(m1 + 2)/2;
//  nb2 = (m2 + 1)*(m2 + 2)/2;
//  nb3 = (m3 + 1)*(m3 + 2)/2;
//  nb4 = (m4 + 1)*(m4 + 2)/2;
//  int N=nb1*nb2*nb3*nb4;
//
//  double *eri= new double[N];
//  compute_eri(p1, c1, n1, e1, m1, x1, y1, z1,
//              p2, c2, n2, e2, m2, x2, y2, z2,
//              p3, c3, n3, e3, m3, x3, y3, z3,
//              p4, c4, n4, e4, m4, x4, y4, z4,
//              eri);
////  for (int i=0; i<N; i++) 
////      std::cout << eri[i];
//
//  return 0;
//}

