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
    tors = []
    centers = [ [], [], [], [], [] ]
    for i in range(natoms):
        for j in range(i+1,natoms):
            if A[i,j] == 1:
                conn.append([i,j])    
        if sum(A[i]) == 2:
            centers[0].append( len(angs) )
            a = bondto[i][0]
            b = bondto[i][1]
            p = conn.index([min(a,i),max(a,i)])
            q = conn.index([min(b,i),max(b,i)])
            angs.append( ( a, i, b, p, q ) )
        if sum(A[i]) == 3:
            centers[1].append( len(oops) )
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
            centers[2].append( ( a, a+1, a+2, a+3, a+4, a+5 ) )
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
            a = len(angs)
            centers[4].append( ( a, a+1, a+2, a+3, a+4, a+5, a+6, a+7, a+8, a+9, a+10, a+11, a+12, a+13, a+14 ) )
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
    return natoms, conn, angs, oops, tors, centers

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

def define_internals(X,H,A,bonds,angs,oops):
    X_i = []
    nbonds = len(bonds)
    nangs = len(angs)
    noops = len(oops)
    natoms = len(X)
    nints = nbonds + nangs + noops
    B = np.zeros((nints,3*natoms))
    for p in range(nbonds):
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
                B = moveatom( B, p, j, AB / 2 )
            for j in R:
                B = moveatom( B, p, j, -AB / 2 )
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
        if sum(A[b]) == 3:
            scale = 2 / 3
        else:
            scale = 1
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
                BI, bi = bondvec(X,b,i)
                B = moveatom( B, nr, i, scale * bi * PA / 2 )
            for i in frag_c:
                BI, bi = bondvec(X,b,i)
                B = moveatom( B, nr, i, scale * bi * PC / 2 )
        X_i.append( theta )
    for s in range(noops):
        ns = nbonds + nangs + s
        a, b, c, d, p, q, r = oops[s]
        AB, ab = bondvec(X,b,a)
        AC, ac = bondvec(X,c,a)
        AD, ad = bondvec(X,d,a)
        theta_BAC = angle(AB,AC)
        phi = math.asin(  np.dot( np.cross(AB,AC), AD ) / math.sin(theta_BAC)  )
        if False:
            print( math.asin(  np.dot( np.cross(AB,AC), AD ) / math.sin(angle(AB,AC))  ) )
            print( math.asin(  np.dot( np.cross(AC,AD), AB ) / math.sin(angle(AC,AD))  ) )
            print( math.asin(  np.dot( np.cross(AD,AB), AC ) / math.sin(angle(AD,AB))  ) )
            print( angle(AB,AC) ) 
            print( angle(AC,AD) ) 
            print( angle(AD,AB) ) 
        if sum(A[b]) == 1 and sum(A[c]) == 1 and sum(A[d]) == 1:
            BC, bc = bondvec(X,c,b)
            BD, bd = bondvec(X,d,b)
            planarnorm = normal(BC,BD) 
            b_B = 0.25 * ab * normal(AB,normal(AB,planarnorm))  
            b_C = 0.25 * ac * normal(AC,normal(AC,planarnorm))  
            b_D = 0.25 * ad * normal(AD,normal(AD,planarnorm))  
            b_A = -b_B - b_C - b_D
            if False:
                print(b_D, la.norm(b_D) )
                print(b_B, la.norm(b_B) )
                print(b_C, la.norm(b_C) )
            B = moveatom( B, ns, a, b_A  )
            B = moveatom( B, ns, b, b_B  )
            B = moveatom( B, ns, c, b_C  )
            B = moveatom( B, ns, d, b_D  )
            if False:
                print( "N", X[a,0], X[a,1], X[a,2] )
                print( "H", X[b,0], X[b,1], X[b,2] )
                print( "F", X[b,0]+b_B[0], X[b,1]+b_B[1], X[b,2]+b_B[2] )
                print( "H", X[c,0], X[c,1], X[c,2] )
                print( "F", X[c,0]+b_C[0], X[c,1]+b_C[1], X[c,2]+b_C[2] )
                print( "H", X[d,0], X[d,1], X[d,2] )
                print( "F", X[d,0]+b_D[0], X[d,1]+b_D[1], X[d,2]+b_D[2] )
        elif False:
            b_D = ( np.cross(AB,AC) / math.cos(phi) / math.sin( angle(AB,AC) ) - AD * math.tan(phi) ) / ad
            b_B = ( np.cross(AC,AD) / math.cos(phi) / math.sin( angle(AC,AD) ) - AB * math.tan(phi) ) / ab
            b_C = ( np.cross(AD,AB) / math.cos(phi) / math.sin( angle(AD,AB) ) - AC * math.tan(phi) ) / ac
            b_A = -b_D - b_C - b_B
            print(b_D, la.norm(b_D) )
            print(b_B, la.norm(b_B) )
            print(b_C, la.norm(b_C) )
            print(b_A)
            B = moveatom( B, ns, a, b_A  )
            B = moveatom( B, ns, b, b_B  )
            B = moveatom( B, ns, c, b_C  )
            B = moveatom( B, ns, d, b_D  )
        elif False:
            sinsq = math.sin(theta_BAC)**2
            b_D = ( np.cross(AB,AC) / math.cos(phi) / math.sin(theta_BAC) - AD * math.tan(phi) ) / ad
            b_B = ( np.cross(AC,AD) / math.cos(phi) / math.sin(theta_BAC) - math.tan(phi) / sinsq * ( AB - math.cos(theta_BAC) * AC ) ) / ab 
            b_C = ( np.cross(AD,AB) / math.cos(phi) / math.sin(theta_BAC) - math.tan(phi) / sinsq * ( AC - math.cos(theta_BAC) * AB ) ) / ac 
            b_A = -b_D - b_C - b_B
            print(b_D, la.norm(b_D) )
            print(b_B, la.norm(b_B) )
            print(b_C, la.norm(b_C) )
            print(b_A)
            B = moveatom( B, ns, a, b_A  )
            B = moveatom( B, ns, b, b_B  )
            B = moveatom( B, ns, c, b_C  )
            B = moveatom( B, ns, d, b_D  )
        X_i.append( phi )
    H_i = np.dot(np.dot(B,H),np.transpose(B))
    return X_i, H_i

