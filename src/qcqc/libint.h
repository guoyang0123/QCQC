/*
 *  Copyright (C) 1996-2017 Edward F. Valeev and Justin T. Fermann
 *
 *  This file is part of Libint.
 *
 *  Libint is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Libint is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with Libint.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef _psi3_libint_h
#define _psi3_libint_h

/* Maximum angular momentum of functions in a basis set plus 1 */
#define REALTYPE double
#define LIBINT_MAX_AM 7
#define LIBINT_OPT_AM 4
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
  } prim_data;

typedef struct {
  REALTYPE *int_stack;
  prim_data *PrimQuartet;
  REALTYPE AB[3];
  REALTYPE CD[3];
  REALTYPE *vrr_classes[13][13];
  REALTYPE *vrr_stack;
  } Libint_t;

#ifdef __cplusplus
extern "C" {
#endif
extern REALTYPE *(*build_eri[7][7][7][7])(Libint_t *, int);
void init_libint_base();
int  init_libint(Libint_t *, int max_am, int max_num_prim_comb);
void free_libint(Libint_t *);
int  libint_storage_required(int max_am, int max_num_prim_comb);
#ifdef __cplusplus
}
#endif

#endif

double* init_array(int size);

void free_array(double* array);

void compute_eri(int nprim1, double *c1, double *n1, double *e1, int am1, double x1, double y1, double z1,
                 int nprim2, double *c2, double *n2, double *e2, int am2, double x2, double y2, double z2,
                 int nprim3, double *c3, double *n3, double *e3, int am3, double x3, double y3, double z3,
                 int nprim4, double *c4, double *n4, double *e4, int am4, double x4, double y4, double z4,
                 double *eri);
int compute_pqrs(int p, int q, int r, int s, int n1, int n2, int n3, int n4,
                 bool swap12, bool swap34, bool s13s24);

