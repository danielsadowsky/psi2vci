import argparse
import math
import numpy as np
from scipy import optimize 
import sympy as sym
import read
import write
import datetime

np.set_printoptions(precision=5,suppress=1e-5)

#DEFINE FUNCTIONS:
def eval_eq(function,variables,values,n=3):
    V = function
    for i in range(n):
        V = V.subs(variables[i],values[i])
    return V

# READ PARAMETERS FROM COMMAND LINE:
parser = argparse.ArgumentParser()
parser.add_argument('-P', action='store_true', help='planar model (for bent triatomics)')
parser.add_argument('-T', action='store_true', help='out-of-plane model (for bent triatomics)')
parser.add_argument('-O', action='store_true', help='harmonic bonding potential')
parser.add_argument('-M', action='store_true', help='Morse bonding potential')
parser.add_argument('-V', action='store_true', help='Varshni bonding potential')
parser.add_argument('-C', action='store_true', help='harmonic triatomic potential')
parser.add_argument('-n', type=int, default = 0, help='modified n-Anderson potential')
parser.add_argument('-x', type=str, help='geometry file')
parser.add_argument('-y', type=str, help='Hessian file')
parser.add_argument('-t', type=float, help='total atomization energy (a.u.)')
parser.add_argument('-o', type=str, help='output file')
parser.add_argument('-v', type=int, default=3, help='level of verbose output')
args = parser.parse_args()
verbose = args.v 

# CHECK FOR EXECUTION ERRORS   
if args.O + args.M + args.V != 1:
    print("Choose HO, Morse, or Varshni!")
    exit()
if args.n + args.C == 0:
    print("Choose HO or modified Anderson!")
    exit()

# CREATE LIST OF RULES 
print()
notes = []
notes.append( "GEOMETRY FROM " + args.x ) 
notes.append( "HESSIAN FROM " + args.y ) 
notes.append( "TOTAL ATOMIZATION ENERGY OF " + str(args.t) )
if args.M:
    notes.append("MORSE BONDING POTENTIAL")
elif args.V:
    notes.append("VARSHNI BONDING POTENTIAL")
elif args.O:
    notes.append("HARMONIC BONDING POTENTIAL")
if args.C:
    notes.append("HARMONIC TRIATOMIC POTENTIAL")
elif args.n > 0:
    notes.append("MODIFIED ANDERSON-"+str(args.n)+" TRIATOMIC POTENTIAL")
if args.T:
    notes.append("TETRAHEDRAL LONE PAIR DEFINITION")
elif args.P:
    notes.append("PLANAR LONE PAIR DEFINITION")
if verbose > 1:
    for note in notes:
        print(note)
    print()

if True:
    TAE = args.t 
    Z, X = read.xyz(args.x) 
    H = read.hess(args.y) 
    A = read.adj(args.x)
    natoms, bonds, angs, nops, oops, nors, tors = read.defconn(A)
    nbonds = len(bonds)
    nangs = len(angs)
    noops = len(oops)
    X_i, H_i = read.define_internals(X,H,A,bonds,angs,oops)
    BDE = TAE / nbonds  
if verbose > 2:
    print("SETTING UP INTERNAL COORDINATES:")
    print()
    print("bonds [a.u.]:")
    for i in range(nbonds):
        print( Z[bonds[i][0]], Z[bonds[i][1]], X_i[i] )
    print()
    print("bond angles [rad]:")
    for i in range(nangs):
        print( Z[angs[i][0]], Z[angs[i][1]], Z[angs[i][2]], X_i[nbonds+i] )
    print()
    print("out-of-plane angles [rad]:")
    for i in range(noops):
        print( Z[oops[i][0]], Z[oops[i][1]], Z[oops[i][2]], Z[oops[i][3]], X_i[nbonds+nangs+i] )
    print()
    print("Hessian [a.u.]:")
    print(H_i)
    print()
    print("bond dissociation energies [a.u.]:")
    for i in range(nbonds):
        print(BDE)
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
    for i in range(noops):
        p = oops[i][4]
        q = oops[i][5]
        r = oops[i][6]
        for j in range(nangs):
            a = angs[j][3]
            b = angs[j][4]
            if min(a,b) == q and max(a,b) == r:  
                t_p = nbonds + j 
            elif min(a,b) == p and max(a,b) == r:  
                t_q = nbonds + j 
            elif min(a,b) == p and max(a,b) == q:  
                t_r = nbonds + j 
        s = nbonds + nangs + i
        if args.C:
            # HARMONIC TETRATOMIC
            G += 0.5 * H_i[s,s] * ( X_S[s] - X_i[s] )**2 
            G += H_i[p,t_p] * ( X_S[p] - X_i[p] ) * ( X_S[t_p] - X_i[t_p] )
            G += H_i[q,t_q] * ( X_S[q] - X_i[q] ) * ( X_S[t_q] - X_i[t_q] )
            G += H_i[r,t_r] * ( X_S[r] - X_i[r] ) * ( X_S[t_r] - X_i[t_r] )
            G += H_i[t_p,t_q] * ( X_S[t_p] - X_i[t_p] ) * ( X_S[t_q] - X_i[t_q] )
            G += H_i[t_p,t_r] * ( X_S[t_p] - X_i[t_p] ) * ( X_S[t_r] - X_i[t_r] )
            G += H_i[t_q,t_r] * ( X_S[t_q] - X_i[t_q] ) * ( X_S[t_r] - X_i[t_r] )
            G += H_i[p,s] * ( X_S[p] - X_i[p] ) * ( X_S[s] - X_i[s] )
            G += H_i[q,s] * ( X_S[q] - X_i[q] ) * ( X_S[s] - X_i[s] )
            G += H_i[r,s] * ( X_S[r] - X_i[r] ) * ( X_S[s] - X_i[s] )
            G += H_i[t_p,s] * ( X_S[t_p] - X_i[t_p] ) * ( X_S[s] - X_i[s] )
            G += H_i[t_q,s] * ( X_S[t_q] - X_i[t_q] ) * ( X_S[s] - X_i[s] )
            G += H_i[t_r,s] * ( X_S[t_r] - X_i[t_r] ) * ( X_S[s] - X_i[s] )
    for i in range(nangs): 
        p = angs[i][3]
        q = angs[i][4] 
        r = nbonds + i
        R_V = ( X_S[p] * X_S[p] + X_S[q] * X_S[q] - 2 * X_S[p] * X_S[q] * sym.cos(X_S[r]) )**(0.5)
        R_E = sym.sqrt( X_i[p] * X_i[p] + X_i[q] * X_i[q] - 2 * X_i[p] * X_i[q] * sym.cos(X_i[r]) )
        if args.P:
            R_L = sym.sqrt( X_S[p] * X_S[p] + X_S[q] * X_S[q] - 2 * X_S[p] * X_S[q] * sym.cos( math.pi - 0.5 * X_S[r]) )
        elif args.T:
            R_L = sym.sqrt( ( X_S[p] * sym.sin( X_S[r] / 2 ) )**2 + ( X_S[p] * sym.cos( X_S[r] / 2 ) + X_S[q] * sym.cos( X_S[r] / 2 ) )**2 + ( X_S[q] * sym.sin( X_S[r] / 2 ) )**2 )
        if True:
            # STRICT SEPARATION OF POTENTIALS:
            D = 0.5 * TAE
            x_0 = X_i[p]
            U_0 = 0.0
            J = 0.0
            K = 0.0 
        if args.C:
            # HARMONIC TRIATOMIC
            sol = []
            G += H_i[p,q] * ( X_S[p] - X_i[p] ) * ( X_S[q] - X_i[q] ) + 0.5 * H_i[r,r] * ( X_S[r] - X_i[r] )**2 - U_0
            if abs( X_i[r] - math.pi ) > 1e-4:
                G += H_i[p,r] * ( X_S[p] - X_i[p] ) * ( X_S[r] - X_i[r] )
                G += H_i[q,r] * ( X_S[q] - X_i[q] ) * ( X_S[r] - X_i[r] )
        elif args.n > 0:
            n_f = 1
            # MODIFIED ANDERSON TRIATOMIC
            if abs( X_i[r] - math.pi ) < 1e-4:
                    # LINEAR
                    n = args.n
                    def anderson(parameters,optimization=True):
                        U_anderson = abs(parameters[0]) * R_V**(-n) 
                        HOT = ( X_S[p] * X_S[q] )**(-n_f) 
                        THOL = X_S[p]**(-n_f)
                        THOR = X_S[q]**(-n_f)
                        U_anderson += parameters[1] * HOT
                        U_anderson += parameters[2] * THOL * HOT
                        U_anderson += parameters[3] * THOR * HOT
                        U_anderson += parameters[4] * THOL * THOR * HOT
                        U_anderson += parameters[5] * THOL**2 * HOT
                        U_anderson += parameters[6] * THOR**2 * HOT
                        if optimization == True:
                            dk_xy = eval_eq( sym.diff( sym.diff( U_anderson, X_S[p] ), X_S[q] ), X_S, X_i ) - H_i[p,q]
                            dk_tt = eval_eq( sym.diff( sym.diff( U_anderson, X_S[r] ), X_S[r] ), X_S, X_i ) - H_i[r,r]
                            u_0 = eval_eq( U_anderson, X_S, X_i ) - U_0
                            j_x = eval_eq( sym.diff( U_anderson, X_S[p] ), X_S, X_i ) - J
                            j_y = eval_eq( sym.diff( U_anderson, X_S[q] ), X_S, X_i ) - J
                            k_x = eval_eq( sym.diff( U_anderson, X_S[p], 2 ), X_S, X_i ) - K
                            k_y = eval_eq( sym.diff( U_anderson, X_S[q], 2 ), X_S, X_i ) - K
                            if verbose > 3:
                                print(parameters,dk_xy,dk_tt,u_0,j_x,j_y,k_x,k_y)
                            return [ dk_xy, dk_tt, u_0, j_x, j_y, k_x, k_y ]
                        elif optimization == "U":
                            return U_anderson
                        elif optimization == "dx":
                            return sym.diff( U_anderson, X_S[p] )
                        elif optimization == "dxx":
                            return sym.diff( U_anderson, X_S[p], 2 )
                    alpha = 16*H_i[r,r]*X_i[p]**4
                    #beta = alpha - 16 / 5 * H_i[p,q]*X_i[p]**6
                    #inter = -( alpha + beta ) / ( 2.0 * X_i[p] )**4  
                    guess = [ alpha, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
                    sol = optimize.fsolve( anderson, guess, factor=0.1, xtol=1.0e-8 )
                    if verbose > 1:
                       print("FORCE FIELD PARAMETERS:")
                       print(sol)
                       print()
                    G += anderson(sol,"U") 
            else:
                    # BENT
                    if ( args.P + args.T != 1):
                        print("Choose planar or out-of-plane!")
                        exit()
                    if args.n > 9:
                        n_1 = args.n // 10
                        n_2 = args.n % 10 
                        print(n_1,n_2)
                    else:
                        n_1 = args.n
                        n_2 = args.n
                    def anderson(parameters,optimization=True):
                        U_anderson = abs(parameters[0]) * R_V**(-n_1) 
                        HOT = ( X_S[p] * X_S[q] )**(-n_f) 
                        THOL = X_S[p]**(-n_f)
                        THOR = X_S[q]**(-n_f)
                        U_anderson += parameters[1] * HOT
                        U_anderson += parameters[2] * THOL * HOT
                        U_anderson += parameters[3] * THOR * HOT
                        U_anderson += parameters[4] * THOL * THOR * HOT
                        U_anderson += parameters[5] * THOL**2 * HOT
                        U_anderson += parameters[6] * THOR**2 * HOT
                        U_anderson += ( parameters[7] + parameters[8] * X_S[p] + parameters[9] * X_S[q] ) * R_L**(-n_2)
                        if optimization == True:
                            dk_xy = eval_eq( sym.diff( sym.diff( U_anderson, X_S[p] ), X_S[q] ), X_S, X_i ) - H_i[p,q]
                            dk_tt = eval_eq( sym.diff( sym.diff( U_anderson, X_S[r] ), X_S[r] ), X_S, X_i ) - H_i[r,r]
                            dk_xt = eval_eq( sym.diff( sym.diff( U_anderson, X_S[p] ), X_S[r] ), X_S, X_i ) - H_i[p,r]
                            dk_yt = eval_eq( sym.diff( sym.diff( U_anderson, X_S[q] ), X_S[r] ), X_S, X_i ) - H_i[q,r]
                            u_0 = eval_eq( U_anderson, X_S, X_i ) - U_0
                            j_x = eval_eq( sym.diff( U_anderson, X_S[p] ), X_S, X_i ) - J
                            j_y = eval_eq( sym.diff( U_anderson, X_S[q] ), X_S, X_i ) - J
                            k_x = eval_eq( sym.diff( U_anderson, X_S[p], 2 ), X_S, X_i ) - K
                            k_y = eval_eq( sym.diff( U_anderson, X_S[q], 2 ), X_S, X_i ) - K
                            g_t = eval_eq( sym.diff( U_anderson, X_S[r] ), X_S, X_i ) 
                            return [ dk_xy, dk_tt, u_0, j_x, j_y, k_x, k_y, dk_xt, dk_yt, g_t  ]
                        elif optimization == "U":
                            return U_anderson
                        elif optimization == "dx":
                            return sym.diff( U_anderson, X_S[p] )
                        elif optimization == "dxx":
                            return sym.diff( U_anderson, X_S[p], 2 )
                    alpha = 16*H_i[r,r]*X_i[p]**4
                    guess = [ alpha, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25 * alpha, 0.0, 0.0  ]
                    sol = optimize.fsolve( anderson, guess, factor=0.1, xtol=1.0e-8 )
                    if verbose > 1:
                       print("FORCE FIELD PARAMETERS:")
                       print(sol)
                       print()
                    G += anderson(sol,"U") 
    for p in range(nbonds):
            if args.O:
                k_0 = H_i[p,p] 
                G += 0.5 * k_0 * ( X_S[p] - x_0 )**2 
            elif args.M:
                # MORSE
                x_0 = X_i[p]
                k_0 = H_i[p,p]
                alpha = math.sqrt( 0.5 * k_0 / D ) 
                G += D * ( 1 - sym.exp( -alpha * ( X_S[p] - x_0 ) ) )**2 
            elif args.V:
                x_0 = X_i[p]
                k_0 = H_i[p,p]
                beta = 0.5 * math.sqrt( 0.5 * k_0 / D ) / x_0 - 0.5 / x_0 / x_0
                G += D * ( 1 - x_0 / X_S[p] * sym.exp( -beta * ( X_S[p] * X_S[p] - x_0 * x_0 ) ) )**2 
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

