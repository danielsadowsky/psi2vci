import math
import numpy as np
from numpy import linalg as la

angstrom = 0.52917721067 # angstrom * bohr-1

def notzero(X,tol=1.0e-8):
    if abs(X) > tol:
        return True
    else:
        return False
 
def xyz(filename,unitconv=angstrom):
    file = open(filename,'r')
    line = file.readline()
    natoms = int(line)
    Z = []
    X = np.zeros((natoms,3))
    line = file.readline()
    for i in range(natoms):
        line = file.readline()
        words = line.split()
        Z.append( words[0].lower().capitalize() )
        for k in range(3):
            X[i,k] = float(words[k+1]) / unitconv 
    file.close()
    return Z, X

def hess(filename,unitconv=1.0):
    file = open(filename,'r')
    line = file.readline()
    words = line.split()
    natoms = int(words[0])
    H = np.zeros((3*natoms,3*natoms))
    for j in range(3*natoms):
        for i in range(natoms):
            line = file.readline()
            words = line.split()
            for k in range(3):
                H[3*i+k,j] = float(words[k]) / unitconv
    file.close()
    return H

def hess_alt(filename,unitconv=1.0):
    file = open(filename,'r')
    line = file.readline()
    words = line.split()
    natoms = int(words[0])
    H = np.zeros((natoms,3,natoms,3))
    for j in range(3*natoms):
        for i in range(natoms):
            line = file.readline()
            words = line.split()
            for k in range(3):
                H[i,k,j/3,j%3] = float(words[k]) / unitconv
    file.close()
    return H

def adj(filename):
    file = open(filename,'r')
    natoms = int( file.readline())
    A = np.zeros((natoms,natoms))
    A = A.astype(int)
    file.readline()
    for i in range(natoms):
        file.readline()
    file.readline()
    for line in file:
        words = line.split()
        i = int(words[0]) - 1
        j = int(words[1]) - 1
        if i < 0 or j < 0:
            print("Start numbering atoms at 1!")
            exit()
        A[i,j] = 1
        A[j,i] = 1
    file.close()
    return A 

def createbondindex(A):
    natoms = len(A)
    B = []
    for i in range(natoms):
        l = []
        for j in range(natoms):
            if A[i,j] == 1:
                l.append(j)
        B.append(l)
    return B 

def defconn(A): 
    natoms = len(A)
    bondto = createbondindex(A)
    conn = []
    angs = []
    oops = []
    nops = []
    nors = []
    tors = []
    for i in range(natoms):
        for j in range(i+1,natoms):
            if A[i,j] == 1:
                conn.append([i,j])    
        if sum(A[i]) == 2:
            a = bondto[i][0]
            b = bondto[i][1]
            p = conn.index([min(a,i),max(a,i)])
            q = conn.index([min(b,i),max(b,i)])
            angs.append( ( a, i, b, p, q ) )
        if sum(A[i]) == 3:
            for j in range(3):
                for k in range(j+1,3):
                    terms = bondto[i][:]
                    b = terms.pop(k)
                    a = terms.pop(j)
                    c = terms.pop(0)
                    p = conn.index([min(a,i),max(a,i)])
                    q = conn.index([min(b,i),max(b,i)])
                    r = conn.index([min(c,i),max(c,i)])
                    angs.append( ( a, i, b, p, q ) )
            a = bondto[i][0]
            b = bondto[i][1]
            c = bondto[i][2]
            p = conn.index([min(a,i),max(a,i)])
            q = conn.index([min(b,i),max(b,i)])
            r = conn.index([min(c,i),max(c,i)])
            oops.append( ( i, a, b, c, p, q, r ) )
        if sum(A[i]) == 4:
            a = len(angs)
            nors.append( (a,a+1,a+2,a+3,a+4,a+5) )
            for j in range(4):
                for k in range(j+1,4):
                    terms = bondto[i][:]
                    b = terms.pop(k) 
                    a = terms.pop(j)
                    c = terms.pop(0)
                    d = terms.pop(0)
                    p = conn.index([min(a,i),max(a,i)])
                    q = conn.index([min(b,i),max(b,i)])
                    angs.append( ( a, i, b, p, q ) )
                    for l in range(k+1,4):
                        terms = bondto[i][:]
                        c = terms.pop(l)
                        b = terms.pop(k) 
                        a = terms.pop(j)
                        r = conn.index([min(c,i),max(c,i)])
                        nops.append( ( p, q, r ) )
        if sum(A[i]) == 5:
            for j in range(5):
                for k in range(j+1,5):
                    terms = bondto[i][:]
                    b = terms.pop(k) 
                    a = terms.pop(j)
                    c = terms.pop(2)
                    d = terms.pop(1)
                    e = terms.pop(0)
                    p = conn.index([min(a,i),max(a,i)])
                    q = conn.index([min(b,i),max(b,i)])
                    r = conn.index([min(c,i),max(c,i)])
                    s = conn.index([min(d,i),max(d,i)])
                    t = conn.index([min(e,i),max(e,i)])
                    angs.append( ( a, i, b, p, q ) )
        if sum(A[i]) == 6:
            for j in range(6):
                for k in range(j+1,6):
                    terms = bondto[i][:]
                    b = terms.pop(k) 
                    a = terms.pop(j)
                    c = terms.pop(3)
                    d = terms.pop(2)
                    e = terms.pop(1)
                    f = terms.pop(0)
                    p = conn.index([min(a,i),max(a,i)])
                    q = conn.index([min(b,i),max(b,i)])
                    r = conn.index([min(c,i),max(c,i)])
                    s = conn.index([min(d,i),max(d,i)])
                    t = conn.index([min(e,i),max(e,i)])
                    u = conn.index([min(f,i),max(f,i)])
                    angs.append( ( a, i, b, p, q ) )
    for q in range(len(conn)):
        i = conn[q][0]
        j = conn[q][1] 
        if sum(A[i]) > 1 and sum(A[j]) > 1:
            ti = bondto[i][:]
            tj = bondto[j][:]
            ti.remove(j)
            tj.remove(i) 
            for k in ti:
                for l in tj:
                    #p = conn.index([min(i,k),max(i,k)])
                    #r = conn.index([min(j,l),max(j,l)])
                    tors.append( ( k, i, j, l ) )
    for x in range(len(nops)):
        ain = []
        for y in range(len(angs)):
            if angs[y][3] in nops[x] and angs[y][4] in nops[x]:
                ain.append(y)
        nops[x] = ( tuple(nops[x]) + tuple(ain) )
    return natoms, conn, angs, nops, oops, nors, tors

def distance_matrix(A):
    D = A.copy()
    natoms = len(A)
    bondto = createbondindex(A) 
    for a in range(natoms):
        well = list(range(natoms))
        well.remove(a)
        sick = [a]
        dead = []
        year = 0
        while len(well) > 0:
            year = year + 1
            new = []
            for i in sick:
                for j in bondto[i]:
                    if j in well:
                        new.append(j)
                        well.remove(j)
                        D[a,j] = year 
                        D[j,a] = year 
            dead.extend(sick)
            sick = new[:] 
    return D

def isdone(D):
    n = len(D)
    for i in range(n):
        for j in range(i+1,n):
            if D[i,j] == 0:
                return False
    return True

def divide(A,a,b):
    natoms = len(A) 
    bondto = createbondindex(A)
    if A[a,b] != 1:
        print("a and b must be bonded to use divide.")
        exit()
    L = [a]
    R = [b]
    while ( len(L) + len(R) + 2 ) < natoms:
        for i in L:
            for j in bondto[i]:
                if ( j != b ) and j not in L:
                    L.append(j)
        for i in R:
            for j in bondto[i]:
                if ( j != a ) and j not in R:
                    R.append(j)
    return L, R

def bondvec(X,a,b,normalized=True):
    V = X[a] - X[b]
    v = la.norm(V)
    V = V / v
    return V, v

def angle(RA,RB):
    return np.arccos(np.clip( np.dot(RA,RB), -1.0, 1.0 ))

def normal(RA,RB):
    if abs( angle(RA,RB) - math.pi ) < 1e-5:
        # LINEAR
        n_seed = np.cross( RA, np.random.rand(3) )
        N = np.cross( RB, n_seed )
    else:
        # BENT
        N = np.cross( RA, RB )
    N = N / la.norm(N)
    return N

def moveatom(B,n,a,v):
    for k in range(3):
        B[n,3*a+k] = v[k]
    return B

def define_internals(X,H,A,bonds,angs):
    X_i = []
    nbonds = len(bonds)
    nangs = len(angs)
    natoms = len(X)
    B = np.zeros((nbonds+nangs,3*natoms))
    for p in range(len(bonds)):
        a = bonds[p][0]
        b = bonds[p][1]
        AB, ab = bondvec(X,a,b)
        if sum(A[a]) == 1:
            B = moveatom( B, p, a, AB )
        elif sum(A[b]) == 1:
            B = moveatom( B, p, b, -AB )
        else:
            L, R = read.divide(A,a,b)
            for j in L:
                B = moveatom( B, p, j, AB )
            for j in R:
                B = moveatom( B, p, j, -AB )
        X_i.append( ab )
    for r in range(nangs):
        nr = nbonds + r
        a, b, c, p, q = angs[r]
        AB, ab = bondvec(X,a,b)
        CB, cb = bondvec(X,c,b)
        theta = angle(AB,CB)
        norm = normal(AB,CB)
        PA = np.cross( AB, norm )
        PC = -np.cross( CB, norm )
        if sum(A[b]) == 4:
            scale = 5.0 / 6
        elif sum(A[b]) == 3:
            scale =  2.0 / 3
        else:
            scale = 1.0
        if sum(A[a]) == 1 and sum(A[c]) == 1:
            B = moveatom( B, nr, a, scale * ab * PA / 2 )
            B = moveatom( B, nr, c, scale * cb * PC / 2 )
        elif sum(A[a]) == 1 and sum(A[c]) > 1:
            B = moveatom( B, nr, a, scale * ab * PA )
        elif sum(A[c]) == 1 and sum(A[a]) > 1:
            B = moveatom( B, nr, c, scale * cb * PC )
        else:
            frag_a, rest = read.divide(A,a,b)
            rest, frag_c = read.divide(A,b,c)
            for i in frag_a:
                B = moveatom( B, nr, i, scale * ab * PA / 2 )
            for i in frag_c:
                B = moveatom( B, nr, i, scale * cb * PC / 2 )
        X_i.append( theta )
    H_i = np.dot(np.dot(B,H),np.transpose(B))
    return X_i, H_i

