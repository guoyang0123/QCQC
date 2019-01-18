from numba.extending import get_cython_function_address
from numba import njit
import ctypes
import numpy as np
from math import exp,log

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

# signature is:
# Dd_number_t hyp1f1(Dd_number_t a, Dd_number_t b, Dd_number_t x) nogil
# bind to the real space variant of the function
addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1hyp1f1")
functype = ctypes.CFUNCTYPE(_dble, _dble, _dble, _dble)
hyp1f1_float64_fn = functype(addr)

@njit
def hyp1f1_in_njit(a, b, x):    
    return hyp1f1_float64_fn(a, b, x)

@njit
def Fgamma(m,x):
    SMALL=0.00000001
    x = max(abs(x),SMALL)
    val = gamm_inc(m+0.5,x)
    return 0.5*pow(x,-m-0.5)*val;

@njit
def gammln(x):
    cof = np.array([76.18009172947146,-86.50532032941677,
                    24.01409824083091,-1.231739572450155,
                    0.1208650973866179e-2,-0.5395239384953e-5])
    y=x
    tmp=x+5.5
    tmp = tmp - (x+0.5)*log(tmp)
    ser=1.000000000190015 # don't you just love these numbers?!
    for j in range(6):
        y = y+1
        ser = ser+cof[j]/y
    return -tmp+log(2.5066282746310005*ser/x);

@njit
def gamm_inc(a,x):
    gammap,gln = gammp(a,x)
    return exp(gln)*gammap

@njit
def gammp(a,x):
    "Returns the incomplete gamma function P(a;x). NumRec sect 6.2."
    assert (x > 0 and a >= 0), "Invalid arguments in routine gammp"

    if x < (a+1.0): #Use the series representation
        gamser,gln = _gser(a,x)
        return gamser,gln
    #Use the continued fraction representation
    gammcf,gln = _gcf(a,x)
    return 1.0-gammcf ,gln  #and take its complement.

@njit
def _gser(a,x):
    ITMAX=100
    EPS=3.e-7

    gln=gammln(a)
    assert(x>=0),'x < 0 in gser'
    if x == 0 : return 0,gln

    ap = a
    delt = sum = 1./a
    for i in range(ITMAX):
        ap=ap+1.
        delt=delt*x/ap
        sum=sum+delt
        if abs(delt) < abs(sum)*EPS: break
    else:
        print('a too large, ITMAX too small in gser')
    gamser=sum*exp(-x+a*log(x)-gln)
    return gamser,gln

@njit
def _gcf(a,x):
    "Continued fraction representation of Gamma. NumRec sect 6.1"
    ITMAX=100
    EPS=3.e-7
    FPMIN=1.e-30

    gln=gammln(a)
    b=x+1.-a
    c=1./FPMIN
    d=1./b
    h=d
    for i in range(1,ITMAX+1):
        an=-i*(i-a)
        b=b+2.
        d=an*d+b
        if abs(d) < FPMIN: d=FPMIN
        c=b+an/c
        if abs(c) < FPMIN: c=FPMIN
        d=1./d
        delt=d*c
        h=h*delt
        if abs(delt-1.) < EPS: break
    else:
        print('a too large, ITMAX too small in gcf')
    gammcf=exp(-x+a*log(x)-gln)*h
    return gammcf,gln

