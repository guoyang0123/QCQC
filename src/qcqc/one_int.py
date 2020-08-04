import numpy as np
import sys
import copy
sys.setrecursionlimit(100000)
from numba import njit,jit
from timeit import default_timer as timer
#import qcqc
from util import Fgamma,hyp1f1_in_njit

@njit('int64(int64)')
def shell_to_basis(am):
    return (am+1)*(am+2)/2

@njit('int64(int64)')
def fact2(n):
    if (n <= 1): 
        return 1
    return n*fact2(n-2)

@njit('float64[:](int64[:],float64[:])')
def normalize(m,e):
    ''' Routine to normalize the prim functions.
    '''
    mx, my, mz = m
    L = mx+my+mz
    # n is a list of length equal to number primitives
    n = np.sqrt(np.power(2,2*(L)+1.5)*
                np.power(e,L+1.5)/
                fact2(2*mx-1)/fact2(2*my-1)/
                fact2(2*mz-1)/np.power(np.pi,1.5))
    return n

@njit
def boys(n,T):
    return hyp1f1_in_njit(n+0.5,n+1.5,-T)/(2.0*n+1.0)

@njit
def gaussian_product_center(e1,xyz1,e2,xyz2):
    return (e1*xyz1+e2*xyz2)/(e1+e2)

@njit('float64(float64,int64,float64,int64,int64,float64)')
def E(e1,m1,e2,m2,t,r):
    '''Recursive definition of Hermite Gaussian coefficients.
       Returns a float.
       e1, m1 : orbital exponent and angular momentum on Gaussian 1
       e2, m2 : orbital exponent and angular momentum on Gaussian 2
       t: number nodes in Hermite (depends on type of integral, 
          e.g. always zero for overlap integrals)
       r: distance between origins of Gaussian 1 and 2 
    '''
    p = e1 + e2
    q = e1*e2/p
    if (t < 0) or (t > (m1+m2)):
        # out of bounds for t  
        return 0.0
    elif m1 == m2 == t == 0:
        # base case
        return np.exp(-q*r*r) # K_AB
    elif m2 == 0:
        # decrement index m1
        n1 = m1-1       
        return (1/(2*p))*E(e1,n1,e2,m2,t-1,r) - \
                (q*r/e1)*E(e1,n1,e2,m2,t,  r) + \
                   (t+1)*E(e1,n1,e2,m2,t+1,r)
    else:
        # decrement index m2
        n2 = m2-1
        return (1/(2*p))*E(e1,m1,e2,n2,t-1,r) + \
                (q*r/e2)*E(e1,m1,e2,n2,t,  r) + \
                   (t+1)*E(e1,m1,e2,n2,t+1,r)

@njit('float64(float64,int64,float64,int64,int64,float64,float64)')
def EE(e1,m1,e2,m2,t,r,c):
    '''Recursive definition of Hermite Gaussian coefficients.
       Returns a float.
       e1, m1 : orbital exponent and angular momentum on Gaussian 1
       e2, m2 : orbital exponent and angular momentum on Gaussian 2
       t: number nodes in Hermite (depends on type of integral,
          e.g. always zero for overlap integrals)
       r: distance between origins of Gaussian 1 and 2
    '''
    p = e1 + e2
    q = e1*e2/p
    if (t < 0) or (t > (m1+m2)):
        # out of bounds for t
        return 0.0
    elif m1 == m2 == t == 0:
        # base case
        return c # K_AB
    elif m2 == 0:
        # decrement index m1
        n1 = m1-1
        return (1/(2*p))*EE(e1,n1,e2,m2,t-1,r,c) - \
                (q*r/e1)*EE(e1,n1,e2,m2,t,  r,c) + \
                   (t+1)*EE(e1,n1,e2,m2,t+1,r,c)
    else:
        # decrement index m2
        n2 = m2-1
        return (1/(2*p))*EE(e1,m1,e2,n2,t-1,r,c) + \
                (q*r/e2)*EE(e1,m1,e2,n2,t,  r,c) + \
                   (t+1)*EE(e1,m1,e2,n2,t+1,r,c)


@njit
def R(t,u,v,n,CX,CY,CZ,fac):
    '''Returns the Coulomb auxiliary Hermite integrals
       Returns a float.
       Arguments:
       t,u,v:   order of Coulomb Hermite derivative in x,y,z
                (see defs in Helgaker and Taylor)
       n:       order of Boys function
       CX,CY,CZ:Cartesian vector distance between Gaussian
                composite center P and nuclear center C
    '''

    val = 0.0
    if t == u == v == 0:
        val += fac[n]
    elif t == u == 0:
        if v > 1:
            val += (v-1)*R(t,u,v-2,n+1,CX,CY,CZ,fac)
        val += CZ*R(t,u,v-1,n+1,CX,CY,CZ,fac)
    elif t == 0:
        if u > 1:
            val += (u-1)*R(t,u-2,v,n+1,CX,CY,CZ,fac)
        val += CY*R(t,u-1,v,n+1,CX,CY,CZ,fac)
    else:
        if t > 1:
            val += (t-1)*R(t-2,u,v,n+1,CX,CY,CZ,fac)
        val += CX*R(t-1,u,v,n+1,CX,CY,CZ,fac)
    return val

@njit
def overlap(e1,m1,xyz1,e2,m2,xyz2):
    '''Evaluates overlap integral between two Gaussians
       Returns a float.
       e1, m1, xyz1 : info of contracted Gaussian 1
       e2, m2, xyz2 : info of contracted Gaussian 2
    '''
    m1x, m1y, m1z = m1 # shell angular momentum on Gaussian 1
    m2x, m2y, m2z = m2 # shell angular momentum on Gaussian 2

    DX = xyz1[0]-xyz2[0]
    DY = xyz1[1]-xyz2[1]
    DZ = xyz1[2]-xyz2[2]

    SX = E(e1,m1x,e2,m2x,0,DX) # X
    SY = E(e1,m1y,e2,m2y,0,DY) # Y
    SZ = E(e1,m1z,e2,m2z,0,DZ) # Z
    return SX*SY*SZ*np.power(np.pi/(e1+e2),1.5)

@njit
def S(c1, p1, n1, e1, am1, xyz1, c2, p2, n2, e2, am2, xyz2):
    '''Evaluates overlap between two contracted Gaussians
       over all S, P, D, ... functions
       Arguments:
       c1, n1, e1, am1, xyz1 : info of contracted Gaussian 1
       c2, n2, e2, am2, xyz2 : info of contracted Gaussian 2
       p1, and p2 are normalization factor of prim Gaussian
       of first angular momentum.
    '''

    s = np.zeros(shape=(shell_to_basis(am1),shell_to_basis(am2)))

    offi=0
    for k1 in range(am1+1):
        m1x=am1-k1
        for m1z in range(k1+1):
            m1y = k1 - m1z;
            m1  = np.array([m1x, m1y, m1z])
            offj=0
            for k2 in range(am2+1):
                m2x=am2-k2
                for m2z in range(k2+1):
                    m2y = k2 - m2z;
                    m2  = np.array([m2x, m2y, m2z])
                    for i in range(c1.shape[0]):
                        for j in range(c2.shape[0]):
                            s[offi][offj] += p1[i]*p2[j]*c1[i]*c2[j]*\
                                             overlap(e1[i],m1,xyz1,e2[j],m2,xyz2)
                    s[offi][offj]*=n1[offi]*n2[offj]
                    offj+=1
            offi+=1
    return s

@njit
def kinetic(e1,m1,xyz1,e2,m2,xyz2):
    '''Evaluates kinetic energy integral between two Gaussians
       Returns a float.
       Arguments:
       e1, m1, xyz1 : info of contracted Gaussian 1
       e2, m2, xyz2 : info of contracted Gaussian 2
    '''
    m2x, m2y, m2z = m2

    term0 = e2*(2*(m2x+m2y+m2z)+3)*overlap(e1,m1,xyz1,e2,m2,xyz2)

    m2x2   = np.copy(m2)
    m2y2   = np.copy(m2)
    m2z2   = np.copy(m2)

    m2x2[0]= m2x2[0]+2
    m2y2[1]= m2y2[1]+2
    m2z2[2]= m2z2[2]+2

    term1 = -2*np.power(e2,2)*\
                             (overlap(e1, m1, xyz1, e2, m2x2, xyz2) +
                              overlap(e1, m1, xyz1, e2, m2y2, xyz2) +
                              overlap(e1, m1, xyz1, e2, m2z2, xyz2))

    m2x2[0]= m2x2[0]-4
    m2y2[1]= m2y2[1]-4
    m2z2[2]= m2z2[2]-4

    term2 = -0.5*(m2x*(m2x-1)*overlap(e1, m1, xyz1, e2, m2x2, xyz2) +
                  m2y*(m2y-1)*overlap(e1, m1, xyz1, e2, m2y2, xyz2) +
                  m2z*(m2z-1)*overlap(e1, m1, xyz1, e2, m2z2, xyz2))
    return term0+term1+term2

@njit
def T(c1, p1, n1, e1, am1, xyz1, c2, p2, n2, e2, am2, xyz2):
    '''Evaluates kinetic energy between two contracted Gaussians
       over all S, P, D ... functions.
       Arguments:
       c1, n1, e1, am1, xyz1 : info of contracted Gaussian 1
       c2, n2, e2, am2, xyz2 : info of contracted Gaussian 2
       p1, and p2 are normalization factor of prim Gaussian
       of first angular momentum.
    '''
    t = np.zeros(shape=(shell_to_basis(am1),shell_to_basis(am2)))

    offi=0
    for k1 in range(am1+1):
        m1x=am1-k1
        for m1z in range(k1+1):
            m1y = k1 - m1z;
            m1  = np.array([m1x, m1y, m1z])
            offj=0
            for k2 in range(am2+1):
                m2x=am2-k2
                for m2z in range(k2+1):
                    m2y = k2 - m2z;
                    m2  = np.array([m2x, m2y, m2z])
                    for i in range(c1.shape[0]):
                        for j in range(c2.shape[0]):
                            t[offi][offj] += p1[i]*p2[j]*c1[i]*c2[j]*\
                                             kinetic(e1[i],m1,xyz1,e2[j],m2,xyz2)
                    t[offi][offj]*=n1[offi]*n2[offj]
                    offj+=1
            offi+=1
    return t

@njit
def nuclear_attraction(e1,m1,xyz1,e2,m2,xyz2,xyz_all,proton_all):
    '''Evaluates nuclear attraction energy integral 
       between two prim Gaussian.
       e1, m1, xyz1 : info of contracted Gaussian 1
       e2, m2, xyz2 : info of contracted Gaussian 2
       xyz_all      : the center of all nucleus
       proton_all   : the charge of all nucleus
    '''
    m1x, m1y, m1z = m1 # shell angular momentum on Gaussian 1
    m2x, m2y, m2z = m2 # shell angular momentum on Gaussian 2

    p  = e1 + e2
    D12X = xyz1[0]-xyz2[0]
    D12Y = xyz1[1]-xyz2[1]
    D12Z = xyz1[2]-xyz2[2]

    #Being tested a few times, there is no need to pre-calculate
    #The E elememts. However, I leave this part here
    #maxm  = np.max(np.array([m1x+m2x+1,m1y+m2y+1,m1z+m2z+1]))
    #EXYZ  = np.zeros(shape=(maxm,3));
    #for t in range(maxm):
    #    EXYZ[t][0]=E(e1,m1x,e2,m2x,t,DX)
    #    EXYZ[t][1]=E(e1,m1y,e2,m2y,t,DY)
    #    EXYZ[t][2]=E(e1,m1z,e2,m2z,t,DZ)

    xyz12 = gaussian_product_center(e1,xyz1,e2,xyz2)
    RXYZ  = np.zeros(shape=(m1x+m2x+1,m1y+m2y+1,m1z+m2z+1));
    for i in range(xyz_all.shape[0]):
        D123X = xyz12[0]-xyz_all[i][0]
        D123Y = xyz12[1]-xyz_all[i][1]
        D123Z = xyz12[2]-xyz_all[i][2]
        D123  = np.linalg.norm(xyz12-xyz_all[i])
        T     = D123*D123*p
        fac   = np.zeros(shape=(m1x+m2x+m1y+m2y+m1z+m2z+1));

        for t in range(fac.shape[0]):
            fac[t]=np.power(-2*p,t)*Fgamma(t,T) 
            #The hyp1f1 in scipy is slower than hand code from PyQuante.
            #Leaving here for comparison
            #fac[t]=np.power(-2*p,t)*boys(t,T)

        for t in range(m1x+m2x+1):
            for u in range(m1y+m2y+1):
                for v in range(m1z+m2z+1):
                    RXYZ[t][u][v] +=proton_all[i]*R(t,u,v,0,D123X,D123Y,D123Z,fac)
   
    val = 0.0
    for t in range(m1x+m2x+1):
        SX=E(e1,m1x,e2,m2x,t,D12X)
        for u in range(m1y+m2y+1):
            SY=E(e1,m1y,e2,m2y,u,D12Y)
            for v in range(m1z+m2z+1):
                SZ=E(e1,m1z,e2,m2z,v,D12Z)
                val += SX*SY*SZ*RXYZ[t][u][v]
    val *= -2*np.pi/p 
    return val

@njit
def V(c1, p1, n1, e1, am1, xyz1, c2, p2, n2, e2, am2, xyz2, xyz_all, proton_all):
    '''Evaluates V between two contracted Gaussians
       over all S, P, D ... functions.
       Arguments:
       c1, n1, e1, am1, xyz1 : info of contracted Gaussian 1
       c2, n2, e2, am2, xyz2 : info of contracted Gaussian 2
       xyz_all   : the center of all nucleus
       proton_all: the charge of all nucleus
       p1, and p2 are normalization factor of prim Gaussian
       of first element in the angular momentum.
    '''

    v = np.zeros(shape=(shell_to_basis(am1),shell_to_basis(am2)))

    offi=0
    for k1 in range(am1+1):
        m1x=am1-k1
        for m1z in range(k1+1):
            m1y = k1 - m1z;
            m1  = np.array([m1x, m1y, m1z])
            offj=0
            for k2 in range(am2+1):
                m2x=am2-k2
                for m2z in range(k2+1):
                    m2y = k2 - m2z;
                    m2  = np.array([m2x, m2y, m2z])
                    for i in range(c1.shape[0]):
                        for j in range(c2.shape[0]):
                            v[offi][offj] += p1[i]*p2[j]*c1[i]*c2[j]*\
                                             nuclear_attraction(e1[i],m1,xyz1,e2[j],m2,xyz2,xyz_all, proton_all)
                    v[offi][offj]*=n1[offi]*n2[offj]
                    offj+=1
            offi+=1
    return v

@njit
def electron_repulsion(e1,m1,xyz1,e2,m2,xyz2,e3,m3,xyz3,e4,m4,xyz4):
    '''Evaluates two electron integral between four Gaussians
       Returns a float.
       e1, m1, xyz1 : info of contracted Gaussian 1
       e2, m2, xyz2 : info of contracted Gaussian 2
       e3, m3, xyz3 : info of contracted Gaussian 3
       e4, m4, xyz4 : info of contracted Gaussian 4

    '''
    m1x, m1y, m1z = m1 # shell angular momentum on Gaussian '1'
    m2x, m2y, m2z = m2 # shell angular momentum on Gaussian '2'
    m3x, m3y, m3z = m3 # shell angular momentum on Gaussian '3'
    m4x, m4y, m4z = m4 # shell angular momentum on Gaussian '4'

    D12X = xyz1[0]-xyz2[0]
    D12Y = xyz1[1]-xyz2[1]
    D12Z = xyz1[2]-xyz2[2]

    D34X = xyz3[0]-xyz4[0]
    D34Y = xyz3[1]-xyz4[1]
    D34Z = xyz3[2]-xyz4[2]

    p = e1+e2 # composite exponent for P (from Gaussians 'a' and 'b')
    q = e3+e4 # composite exponent for Q (from Gaussians 'c' and 'd')
    alpha = p*q/(p+q)
    P = gaussian_product_center(e1,xyz1,e2,xyz2) # A and B composite center
    Q = gaussian_product_center(e3,xyz3,e4,xyz4) # C and D composite center
    DPQX = P[0]-Q[0]
    DPQY = P[1]-Q[1]
    DPQZ = P[2]-Q[2]
    DPQ  = np.linalg.norm(P-Q)

    fac0 = -(e1*e2)/(e1+e2)
    fac1 = -(e3*e4)/(e3+e4)
    fac12= np.exp(fac0*D12X*D12X) 
    fac34= np.exp(fac1*D34X*D34X)

    LX   = m1x+m2x+m3x+m4x+1
    ETW  = np.zeros(shape=LX)
    for t in range(m1x+m2x+1):
        ET=EE(e1,m1x,e2,m2x,t,D12X,fac12)
        for w in range(m3x+m4x+1):
            ETW[t+w]+=ET*EE(e3,m3x,e4,m4x,w,D34X,fac34)*np.power(-1,w)

    fac12= np.exp(fac0*D12Y*D12Y)
    fac34= np.exp(fac1*D34Y*D34Y)
    LY   = m1y+m2y+m3y+m4y+1    
    EUX  = np.zeros(shape=LY)
    for u in range(m1y+m2y+1):
        EU=EE(e1,m1y,e2,m2y,u,D12Y,fac12)
        for x in range(m3y+m4y+1):
            EUX[u+x]+=EU*EE(e3,m3y,e4,m4y,x,D34Y,fac34)*np.power(-1,x)

    fac12= np.exp(fac0*D12Z*D12Z)
    fac34= np.exp(fac1*D34Z*D34Z)
    LZ   = m1z+m2z+m3z+m4z+1
    EVY  = np.zeros(shape=LZ)
    for v in range(m1z+m2z+1):
        EV=EE(e1,m1z,e2,m2z,v,D12Z,fac12)
        for y in range(m3z+m4z+1):
            EVY[v+y]+=EV*EE(e3,m3z,e4,m4z,y,D34Z,fac34)*np.power(-1,y)

    T    = alpha*DPQ*DPQ
    fac  = np.zeros(shape=(m1x+m2x+m3x+m4x+m1y+m2y+m3y+m4y+m1z+m2z+m3z+m4z+1))
    for t in range(fac.shape[0]):
        fac[t]=np.power(-2*alpha,t)*Fgamma(t,T)

    val = 0.0
    for t in range(LX):        
        for u in range(LY):
            for v in range(LZ):
                val += ETW[t]*EUX[u]*EVY[v]*R(t,u,v,0,DPQX,DPQY,DPQZ,fac)

    val *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q))
    return val

@njit
def ERI(c1, p1, n1, e1, am1, xyz1, c2, p2, n2, e2, am2, xyz2,
        c3, p3, n3, e3, am3, xyz3, c4, p4, n4, e4, am4, xyz4):
    '''Evaluates two electron integral between four 
       contracted Gaussians
       Returns a float.
       c1, n1, e1, am1, xyz1 : info of contracted Gaussian 1
       c2, n2, e2, am2, xyz2 : info of contracted Gaussian 2
       c3, n3, e3, am3, xyz3 : info of contracted Gaussian 3
       c4, n4, e4, am4, xyz4 : info of contracted Gaussian 4
       p1, and p2 are normalization factor of prim Gaussian
       of first element in the angular momentum.
    '''
    eri = np.zeros(shape=(shell_to_basis(am1),shell_to_basis(am2),shell_to_basis(am3),shell_to_basis(am4)))
    offi=0
    for k1 in range(am1+1):
        m1x=am1-k1
        for m1z in range(k1+1):
            m1y = k1 - m1z
            m1  = np.array([m1x, m1y, m1z])
            offj=0
            for k2 in range(am2+1):
                m2x=am2-k2
                for m2z in range(k2+1):
                    m2y = k2 - m2z
                    m2  = np.array([m2x, m2y, m2z])
                    offk=0
                    for k3 in range(am3+1):
                        m3x=am3-k3
                        for m3z in range(k3+1):
                            m3y = k3 - m3z
                            m3  = np.array([m3x, m3y, m3z])
                            offl=0
                            for k4 in range(am4+1):
                                m4x=am4-k4
                                for m4z in range(k4+1):
                                    m4y = k4 - m4z
                                    m4  = np.array([m4x, m4y, m4z])
                                    for i in range(c1.shape[0]):
                                        for j in range(c2.shape[0]):
                                            for k in range(c3.shape[0]):
                                                for l in range(c4.shape[0]):
                                                    eri[offi][offj][offk][offl] += \
                                                    p1[i]*p2[j]*p3[k]*p4[l]* c1[i]*c2[j]*c3[k]*c4[l]* \
                                                    electron_repulsion(e1[i], m1, xyz1, e2[j], m2, xyz2,
                                                                       e3[k], m3, xyz3, e4[l], m4, xyz4)
                                    eri[offi][offj][offk][offl]*=n1[offi]*n2[offj]*n3[offk]*n4[offl]
                                    offl+=1
                            offk+=1
                    offj+=1
            offi+=1
    return eri

##############################################################    
#   Below are integral functions interfacing to the cpp lib
##############################################################
@jit
def S_C(c1, p1, n1, e1, am1, xyz1, c2, p2, n2, e2, am2, xyz2):
    '''Evaluates overlap between two contracted Gaussians
       Returns float.
       Arguments:
       c1, n1, e1, am1, xyz1 : info of contracted Gaussian 1
       c2, n2, e2, am2, xyz2 : info of contracted Gaussian 2
       p1, and p2 are normalization factor of prim Gaussian
       of first element in the angular momentum.
    '''
    x1, y1, z1 = xyz1  # xyz of Gaussian 1
    x2, y2, z2 = xyz2  # xyz of Gaussian 2

    s = np.zeros(shape=(shell_to_basis(am1),shell_to_basis(am2)))

    offi=0
    for k1 in range(am1+1):
        m1x=am1-k1
        for m1z in range(k1+1):
            m1y = k1 - m1z;
            offj=0
            for k2 in range(am2+1):
                m2x=am2-k2
                for m2z in range(k2+1):
                    m2y = k2 - m2z;
                    for i in range(c1.shape[0]):
                        for j in range(c2.shape[0]):
                            s[offi][offj] += p1[i]*p2[j]*c1[i]*c2[j]*\
                                             qcqc.overlap(e1[i], m1x, m1y, m1z, x1, y1, z1,
                                                          e2[j], m2x, m2y, m2z, x2, y2, z2)
                    s[offi][offj]*=n1[offi]*n2[offj]
                    offj+=1
            offi+=1
    return s


@jit
def T_C(c1, p1, n1, e1, am1, xyz1, c2, p2, n2, e2, am2, xyz2):
    '''Evaluates kinetic between two contracted Gaussians
       Arguments:
       c1, n1, e1, am1, xyz1 : info of contracted Gaussian 1
       c2, n2, e2, am2, xyz2 : info of contracted Gaussian 2
       p1, and p2 are normalization factor of prim Gaussian
       of first element in the angular momentum.
    '''

    x1, y1, z1 = xyz1  # xyz of Gaussian 1
    x2, y2, z2 = xyz2  # xyz of Gaussian 2

    t = np.zeros(shape=(shell_to_basis(am1),shell_to_basis(am2)))

    offi=0
    for k1 in range(am1+1):
        m1x=am1-k1
        for m1z in range(k1+1):
            m1y = k1 - m1z;
            offj=0
            for k2 in range(am2+1):
                m2x=am2-k2
                for m2z in range(k2+1):
                    m2y = k2 - m2z;
                    for i in range(c1.shape[0]):
                        for j in range(c2.shape[0]):
                            t[offi][offj] += p1[i]*p2[j]*c1[i]*c2[j]*\
                                             qcqc.kinetic(e1[i], m1x, m1y, m1z, x1, y1, z1,
                                                          e2[j], m2x, m2y, m2z, x2, y2, z2)
                    t[offi][offj]*=n1[offi]*n2[offj]
                    offj+=1
            offi+=1
    return t

@jit
def V_C(c1, p1, n1, e1, am1, xyz1, c2, p2, n2, e2, am2, xyz2, xyz3):
    '''Evaluates nuclear attraction matrix 
       between two contracted Gaussians
       Arguments:
       c1, n1, e1, am1, xyz1 : info of contracted Gaussian 1
       c2, n2, e2, am2, xyz2 : info of contracted Gaussian 2
       p1, and p2 are normalization factor of prim Gaussian
       of first element in the angular momentum.
    '''

    x1, y1, z1 = xyz1 # xyz of Gaussian 1
    x2, y2, z2 = xyz2 # xyz of Gaussian 2
    x3, y3, z3 = xyz3 # xyz of Gaussian 3

    v = np.zeros(shape=(shell_to_basis(am1),shell_to_basis(am2)))

    offi=0
    for k1 in range(am1+1):
        m1x=am1-k1
        for m1z in range(k1+1):
            m1y = k1 - m1z;
            offj=0
            for k2 in range(am2+1):
                m2x=am2-k2
                for m2z in range(k2+1):
                    m2y = k2 - m2z;
                    for i in range(c1.shape[0]):
                        for j in range(c2.shape[0]):
                            v[offi][offj] += p1[i]*p2[j]*c1[i]*c2[j]*\
                                             qcqc.nuclear(e1[i], m1x, m1y, m1z, x1, y1, z1,
                                                          e2[j], m2x, m2y, m2z, x2, y2, z2,
                                                                                x3, y3, z3)
                    v[offi][offj]*=n1[offi]*n2[offj]
                    offj+=1
            offi+=1
    return v

@jit
def ERI_C(c1, p1, n1, e1, am1, xyz1, c2, p2, n2, e2, am2, xyz2,
          c3, p3, n3, e3, am3, xyz3, c4, p4, n4, e4, am4, xyz4):
    '''Evaluates two electron integral between four
       contracted Gaussians
       Returns a float.
       c1, n1, e1, am1, xyz1 : info of contracted Gaussian 1
       c2, n2, e2, am2, xyz2 : info of contracted Gaussian 2
       c3, n3, e3, am3, xyz3 : info of contracted Gaussian 3
       c4, n4, e4, am4, xyz4 : info of contracted Gaussian 4
       p1, and p2 are normalization factor of prim Gaussian
       of first element in the angular momentum.
    '''
    x1, y1, z1 = xyz1 # xyz of Gaussian 1
    x2, y2, z2 = xyz2 # xyz of Gaussian 2
    x3, y3, z3 = xyz3 # xyz of Gaussian 3
    x4, y4, z4 = xyz4 # xyz of Gaussian 4

    eri = np.zeros(shape=(shell_to_basis(am1),shell_to_basis(am2),shell_to_basis(am3),shell_to_basis(am4)))

    offi=0
    for k1 in range(am1+1):
        m1x=am1-k1
        for m1z in range(k1+1):
            m1y = k1 - m1z;
            offj=0
            for k2 in range(am2+1):
                m2x=am2-k2
                for m2z in range(k2+1):
                    m2y = k2 - m2z;
                    offk=0
                    for k3 in range(am3+1):
                        m3x=am3-k3
                        for m3z in range(k3+1):
                            m3y = k3 - m3z;
                            offl=0
                            for k4 in range(am4+1):
                                m4x=am4-k4
                                for m4z in range(k4+1):
                                    m4y = k4 - m4z;
                                    for i in range(c1.shape[0]):
                                        for j in range(c2.shape[0]):
                                            for k in range(c3.shape[0]):
                                                for l in range(c4.shape[0]):
                                                    eri[offi][offj][offk][offl] += c1[i]*c2[j]*c3[k]*c4[l]* \
                                                    qcqc.coulomb_repulsion(x1, y1, z1, p1[i], m1x, m1y, m1z, e1[i],
                                                                           x2, y2, z2, p2[j], m2x, m2y, m2z, e2[j],
                                                                           x3, y3, z3, p3[k], m3x, m3y, m3z, e3[k],
                                                                           x4, y4, z4, p4[l], m4x, m4y, m4z, e4[l])
                                    eri[offi][offj][offk][offl]*=n1[offi]*n2[offj]*n3[offk]*n4[offl]
                                    offl+=1
                            offk+=1
                    offj+=1
            offi+=1
    return eri

def ERI_L(c1, p1, n1, e1, am1, xyz1, c2, p2, n2, e2, am2, xyz2,
          c3, p3, n3, e3, am3, xyz3, c4, p4, n4, e4, am4, xyz4):
    '''Evaluates two electron integral between four
       contracted Gaussians
       Returns a float.
       e1, p1, n1, am1, xyz1 : info of contracted Gaussian 1
       e2, p2, n2, am2, xyz2 : info of contracted Gaussian 2
       e3, p3, n3, am3, xyz3 : info of contracted Gaussian 3
       e4, p4, n4, am4, xyz4 : info of contracted Gaussian 4
    '''

    nprim1=c1.shape[0]
    nprim2=c2.shape[0]
    nprim3=c3.shape[0]
    nprim4=c4.shape[0]

    nb1 = shell_to_basis(am1)
    nb2 = shell_to_basis(am2)
    nb3 = shell_to_basis(am3)
    nb4 = shell_to_basis(am4)

    x1, y1, z1 = xyz1 # xyz of Gaussian 1
    x2, y2, z2 = xyz2 # xyz of Gaussian 2
    x3, y3, z3 = xyz3 # xyz of Gaussian 3
    x4, y4, z4 = xyz4 # xyz of Gaussian 4

    tmp=np.zeros(shape=nb1*nb2*nb3*nb4)

    qcqc.compute_eri(nprim1, c1, p1, n1, e1, am1, x1, y1, z1,
                     nprim2, c2, p2, n2, e2, am2, x2, y2, z2,
                     nprim3, c3, p3, n3, e3, am3, x3, y3, z3,
                     nprim4, c4, p4, n4, e4, am4, x4, y4, z4, tmp)

    eri = np.reshape(tmp,(nb1, nb2, nb3, nb4))
    #print(eri)
    return eri

@njit
def dipole(e1,m1,xyz1,e2,m2,xyz2,xyz3,direction):

    m1x, m1y, m1z = m1 # shell angular momentum on Gaussian '1'
    m2x, m2y, m2z = m2 # shell angular momentum on Gaussian '2'

    P  = gaussian_product_center(e1,xyz1,e2,xyz2)
    DX = xyz1[0]-xyz2[0]
    DY = xyz1[1]-xyz2[1]
    DZ = xyz1[2]-xyz2[2]

    if direction == 0:
        PQX = P[0] - xyz3[0]
        SX  = E(e1,m1x,e2,m2x,1,DX) + PQX*E(e1,m1x,e2,m2x,0,DX)
        #SX = E(e1,m1x,e2,m2x,1,DX) # X
        SY  = E(e1,m1y,e2,m2y,0,DY) # Y
        SZ  = E(e1,m1z,e2,m2z,0,DZ) # Z

        return SX*SY*SZ*np.power(np.pi/(e1+e2),1.5)

    elif direction == 1:
        PQY = P[1] - xyz3[1]
        SX  = E(e1,m1x,e2,m2x,0,DX)# X
        SY  = E(e1,m1y,e2,m2y,1,DY) + PQY*E(e1,m1y,e2,m2y,0,DY)
        #SY = E(e1,m1y,e2,m2y,1,DY) # Y
        SZ  = E(e1,m1z,e2,m2z,0,DZ) # Z

        return SX*SY*SZ*np.power(np.pi/(e1+e2),1.5)

    elif direction == 2:
        PQZ = P[2] - xyz3[2]
        SX  = E(e1,m1x,e2,m2x,0,DX) # X
        SY  = E(e1,m1y,e2,m2y,0,DY) # Y
        SZ  = E(e1,m1z,e2,m2z,1,DZ) + PQZ*E(e1,m1z,e2,m2z,0,DZ)
        #SZ  = E(e1,m1z,e2,m2z,1,DZ) # Z

        return SX*SY*SZ*np.power(np.pi/(e1+e2),1.5)

@njit
def DIP(c1, p1, n1, e1, am1, xyz1, c2, p2, n2, e2, am2, xyz2, xyz3, direction):
    '''Evaluates dipole between two contracted Gaussians
       over all S, P, D ... functions.
       Arguments:
       c1, n1, e1, am1, xyz1 : info of contracted Gaussian 1
       c2, n2, e2, am2, xyz2 : info of contracted Gaussian 2
       xyz3   : the center defined by users
       p1, and p2 are normalization factor of prim Gaussian
       of first element in the angular momentum.
    '''

    d = np.zeros(shape=(shell_to_basis(am1),shell_to_basis(am2)))
    offi=0
    for k1 in range(am1+1):
        m1x=am1-k1
        for m1z in range(k1+1):
            m1y = k1 - m1z;
            m1  = np.array([m1x, m1y, m1z])
            offj=0
            for k2 in range(am2+1):
                m2x=am2-k2
                for m2z in range(k2+1):
                    m2y = k2 - m2z;
                    m2  = np.array([m2x, m2y, m2z])
                    for i in range(c1.shape[0]):
                        for j in range(c2.shape[0]):
                            d[offi][offj] -= p1[i]*p2[j]*c1[i]*c2[j]*\
                                             dipole(e1[i],m1,xyz1,e2[j],m2,xyz2,xyz3,direction)
                    d[offi][offj]*=n1[offi]*n2[offj]
                    offj+=1
            offi+=1
    return d


