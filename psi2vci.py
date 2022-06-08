import argparse
import math
import numpy as np
from scipy import optimize 
import sympy as sym
import read
import write
import models
import datetime

np.set_printoptions(precision=5,suppress=1e-5)
output_precision = 6 

#DEFINE FUNCTIONS:
def eval_eq(function,variables,values):
    n = len(variables)
    V = function
    for i in range(n):
        V = V.subs(variables[i],values[i])
    return V

# READ PARAMETERS FROM COMMAND LINE:
parser = argparse.ArgumentParser()
parser.add_argument('-M', action='store_true', help='Morse bonding potential')
parser.add_argument('-V', action='store_true', help='Varshni bonding potential')
parser.add_argument('-t', type=float, default = -1.0, help='compute BDEs from total atomization energy (a.u.)')
parser.add_argument('-n', type=int, default = 0, help='modified n-Anderson potentials')
parser.add_argument('-T', action='store_true', help='out-of-plane model for dummy atoms')
parser.add_argument('-x', type=str, help='geometry file')
parser.add_argument('-y', type=str, help='Hessian file')
parser.add_argument('-c', type=str, default="no_file", help='connectivity file')
parser.add_argument('-o', type=str, default="output.ff", help='output file')
parser.add_argument('-v', type=int, default=3, help='level of verbose output')
args = parser.parse_args()
verbose = args.v 

# CREATE LIST OF RULES 
print()
notes = []
notes.append( "GEOMETRY FROM " + args.x ) 
notes.append( "HESSIAN FROM " + args.y ) 
if args.t > 0:
    notes.append( "TOTAL ATOMIZATION ENERGY OF " + str(args.t) )
if args.M:
    notes.append("MORSE BONDING POTENTIAL")
elif args.V:
    notes.append("VARSHNI BONDING POTENTIAL")
else:
    notes.append("HARMONIC BONDING POTENTIAL")
if args.n > 0:
    notes.append("MODIFIED ANDERSON-"+str(args.n)+" COORDINATION POTENTIALS")
    if args.T:
        notes.append("TETRAHEDRAL LONE PAIR DEFINITION")
    else:
        notes.append("PLANAR LONE PAIR DEFINITION")
else:
    notes.append("HARMONIC COORDINATION POTENTIALS")
if verbose > 1:
    for note in notes:
        print(note)
    print()

if True:
    Z, X = read.xyz(args.x) 
    H = read.hess(args.y) 
    if args.c == "no_file":
        A = read.find_bonds(X,H)
    else:
        natoms = len(X)
        A, D = read.adj(args.c,natoms)
    natoms, bonds, angs, oops, tors, centers = read.defconn(A)
    nbonds = len(bonds)
    nangs = len(angs) 
    noops = len(oops)
    ntors = len(tors)
    nints = nbonds + nangs + noops + ntors
    X_i, H_i = read.define_internals(X,H,A,bonds,angs,oops)
    if args.t > 0.0:
        D  = [ args.t / nbonds ] * nbonds 
    elif args.c == "no_file":
        D = [ 0.0 ] * nbonds
if verbose > 2:
    print("SETTING UP INTERNAL COORDINATES:")
    print()
    print("bond lengths and dissociation energies [a.u.]:")
    for i in range(nbonds):
        print( i, Z[bonds[i][0]], Z[bonds[i][1]], round(X_i[i],output_precision), round(D[i],output_precision) )
    print()
    print("bond angles [rad]:")
    for i in range(nangs):
        print( nbonds + i, Z[angs[i][0]], Z[angs[i][1]], Z[angs[i][2]], round(X_i[nbonds+i],output_precision) )
    print()
    print("out-of-plane angles [rad]:")
    for i in range(noops):
        print( nbonds + nangs + i, Z[oops[i][0]], Z[oops[i][1]], Z[oops[i][2]], Z[oops[i][3]], round(X_i[nbonds+nangs+i],output_precision) )
    print()
    print("Hessian [a.u.]:")
    print(H_i)
    print()

if True:
    if True:
        # CONSTRUCT POTENTIAL:
        G = 0
        X_S = []
        for i in range(nbonds):
            X_S.append( sym.Symbol( 'r' + str(i) ) ) 
        for i in range(nangs):
            X_S.append( sym.Symbol( 't' + str(i) ) ) 
        for i in range(noops):
            X_S.append( sym.Symbol( 'p' + str(i) ) ) 
    # HEXACOORDINATION POTENTIALS:
    for i in range(len(centers[4])):
        indices = [ nbonds + centers[4][i][j] for j in range(15) ] 
        G = models.hexacoord_harmonic(G,H_i,X_S,X_i,indices,angs)
    # TETRACOORDINATION POTENTIALS:
    for i in range(len(centers[2])):
        indices = [ nbonds + centers[2][i][j] for j in range(6) ] 
        if args.n > 1:
            G = models.tetracoord_tetrahedral(G,H_i,X_S,X_i,indices,angs,args.n)
        else:
            G = models.tetracoord_harmonic(G,H_i,X_S,X_i,indices,angs)
    # TRICOORDINATION POTENTIALS:
    for i in centers[1]:
        p = oops[i][4]
        q = oops[i][5]
        r = oops[i][6]
        s = nbonds + nangs + i
        for j in range(nangs):
            a = angs[j][3]
            b = angs[j][4]
            if min(a,b) == q and max(a,b) == r:  
                t_p = nbonds + j 
            elif min(a,b) == p and max(a,b) == r:  
                t_q = nbonds + j 
            elif min(a,b) == p and max(a,b) == q:  
                t_r = nbonds + j 
        indices = p, q, r, t_p, t_q, t_r, s
        if args.n > 0:
            R_S = []
            for (x,y,t) in [ (p,q,t_r), (p,r,t_q), (q,r,t_p) ]:
                R_S.append( sym.sqrt( X_S[x]**2 + X_S[y]**2 - 2 * X_S[x] * X_S[y] * sym.cos(X_S[t]) ) )
                if args.T:
                    R_S.append( sym.sqrt( (X_S[x] * sym.sin( X_S[t] / 2 ) )**2 + ( X_S[x] * sym.cos( X_S[t] / 2 ) + X_S[y] * sym.cos( X_S[t] / 2 ) )**2 + ( X_S[y] * sym.sin( X_S[t] / 2 ) )**2 ) )
                else:
                    R_S.append( sym.sqrt( X_S[x]**2 + X_S[y]**2 - 2 * X_S[x] * X_S[y] * sym.cos( math.pi - 0.5 * X_S[t] ) ) )
            G = models.tricoord_symmetric_planar(G,H_i,X_S,X_i,indices,R_S,args.n)
        else:
            G = models.tricoord_harmonic(G,H_i,X_S,X_i,indices)
    # BICOORDINATION POTENTIALS:
    for i in centers[0]: 
        p = angs[i][3]
        q = angs[i][4] 
        r = nbonds + i
        indices = p, q, r
        if args.n > 0:
            R_S = []
            for (x,y,t) in [ (p,q,r) ]:
                    R_S.append( sym.sqrt( X_S[x]**2 + X_S[y]**2 - 2 * X_S[x] * X_S[y] * sym.cos(X_S[t]) ) )
                    if args.T:
                        R_S.append( sym.sqrt( (X_S[x] * sym.sin( X_S[t] / 2 ) )**2 + ( X_S[x] * sym.cos( X_S[t] / 2 ) + X_S[y] * sym.cos( X_S[t] / 2 ) )**2 + ( X_S[y] * sym.sin( X_S[t] / 2 ) )**2 ) )
                    else:
                        R_S.append( sym.sqrt( X_S[x]**2 + X_S[y]**2 - 2 * X_S[x] * X_S[y] * sym.cos( math.pi - 0.5 * X_S[t] ) ) )
            if abs( X_i[r] - math.pi ) < 1e-4:
                G = models.bicoord_linear(G,H_i,X_S,X_i,indices,R_S,args.n)
            else:
                G = models.bicoord_bent(G,H_i,X_S,X_i,indices,R_S,args.n)
        else:
            G = models.bicoord_harmonic(G,H_i,X_S,X_i,indices)
    # BONDING POTENTIALS:
    for p in range(nbonds):
            if args.M and D[p] > 0:
                # MORSE
                x_0 = X_i[p]
                k_0 = H_i[p,p]
                alpha = math.sqrt( 0.5 * k_0 / D[p] ) 
                G += D * ( 1 - sym.exp( -alpha * ( X_S[p] - x_0 ) ) )**2 
            elif args.V and D[p] > 0:
                # VARSHNI
                x_0 = X_i[p]
                k_0 = H_i[p,p]
                beta = 0.5 * math.sqrt( 0.5 * k_0 / D[p] ) / x_0 - 0.5 / x_0 / x_0
                G += D[p] * ( 1 - x_0 / X_S[p] * sym.exp( -beta * ( X_S[p] * X_S[p] - x_0 * x_0 ) ) )**2 
            else:
                # HO
                k_0 = H_i[p,p] 
                x_0 = X_i[p]
                G += 0.5 * k_0 * ( X_S[p] - x_0 )**2 
    if verbose > 2 and natoms == 3:
                k_xxx = eval_eq( sym.diff( G, X_S[0], 3), X_S, X_i )   
                k_xxy = eval_eq( sym.diff( sym.diff( G, X_S[0], 2), X_S[1] ), X_S, X_i )   
                k_xxt = eval_eq( sym.diff( sym.diff( G, X_S[0], 2), X_S[2] ), X_S, X_i )   
                k_xyt = eval_eq( sym.diff( sym.diff( sym.diff( G, X_S[0] ), X_S[1] ), X_S[2] ), X_S, X_i )   
                k_xtt = eval_eq( sym.diff( sym.diff( G, X_S[2], 2), X_S[0] ), X_S, X_i )    
                k_ttt = eval_eq( sym.diff( G, X_S[2], 3), X_S, X_i )   
                print("FORCE FIELD THIRD DERIVATIVES:")
                print("k_xxx", k_xxx )
                print("k_xxy", k_xxy )
                print("k_xxt", k_xxt )
                print("k_xyt", k_xyt )
                print("k_xtt", k_xtt )
                print("k_ttt", k_ttt )
                print()
    F = write.calc_tensor(G,X_S,X_i,H_i)
    write.pypes( args.o, F, Z, X, X_i, A, notes )
    print("FORCE FIELD OUTPUT TO " + args.o )

