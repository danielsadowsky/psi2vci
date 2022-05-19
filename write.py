import sys 
import math
import numpy as np
from numpy import linalg as la
from scipy import optimize 
import argparse
import read
#import functions
import sympy as sym
from sympy.solvers.solveset import nonlinsolve

# SET UNITS:
units = [ 1, 1 ]
atto = 4.359744650 # aJ * hartree-1
angstrom = 0.52917721067 # angstrom * bohr-1 

def binom_coefs(C,start=()):
    n_dim = len(start) + 1
    n_var = len(C[1])
    for i in range(n_var):
        coefs = start + (i,)
        C[n_dim][tuple(sorted(list(coefs)))] += 1 
        if (n_dim+1) < len(C):
           C, coefs = binom_coefs(C,coefs)
    return C, coefs

def binom_tensor(nints,order):
    C = [ np.zeros((tuple( [nints] * times ))) for times in range(order+1) ] 
    C, coefs = binom_coefs(C)
    for i in range(order):
        C[i+1][:] /= math.factorial(i+1)
    return C

def calc_derivatives(F,sym_diffs,X_S,X_i,start):
    n = len(start) 
    n_var = len(X_S)
    for i in range(start[-1],n_var):
        coefs = start + (i,)
        diff = sym.diff( sym_diffs[n-1], X_S[i] )
        sym_diffs[n] = diff
        for j in range(n_var):
            #diff = diff.subs(X_S[j],X_0[j])
            diff = diff.subs(X_S[j],X_i[j]) #numerical evaluation to deal with small, near-zero hard-to-factor polynomials of X_0
        F[n][coefs[1:]] = diff 
        if (n+1) < len(F):
            F, sym_diffs, coefs = calc_derivatives(F,sym_diffs,X_S,X_i,coefs) 
    return F, sym_diffs, coefs 

def calc_tensor(G,X_S,X_i,H_i,order=6):
    nints = len(X_S)
    C = binom_tensor(nints,order)
    F = [ np.zeros((tuple( [nints] * times ))) for times in range(order+1) ] 
    sym_diffs = [G] + [0] * order
    F, sym_diffs, final = calc_derivatives(F,sym_diffs,X_S,X_i,(0,))
    for i in range(order+1):
        F[i] *= C[i]
    # CHECK THAT HARMONIC BEHAVIOR IS PRESERVED:
    maxerror = 5.0e-5
    if abs(F[0]) > maxerror:
        print( "ERROR IN HARMONIC BEHAVIOR OF POTENIAL ENERGY SURFACE OTH DERIVATIVE:" )
        exit()
    for i in range(nints):
        if abs(F[1][i]) > maxerror:
            print( "ERROR IN HARMONIC BEHAVIOR OF POTENIAL ENERGY SURFACE 1ST DERIVATIVE:" )
            print( i, " ", F[1][i] )
            exit()
        if abs( F[2][i,i] - 0.5*H_i[i,i] ) > maxerror:
            print( "ERROR IN HARMONIC BEHAVIOR OF POTENIAL ENERGY SURFACE 2ND DERIVATIVE:" )
            print( i, i, F[2][i,i], 0.5*H_i[i,i] )
            exit()
        for j in range(i+1,nints):
            if abs( F[2][i,j] - H_i[i,j] ) > maxerror:
                print( "ERROR IN HARMONIC BEHAVIOR OF POTENIAL ENERGY SURFACE:" )
                print( i, j, F[2][i,j], H_i[i,j] )
                exit()
    # PRINT ANHARMONIC DATA:
    if False:
        print("Setting up anharmonic potential energy surface for method ...")
        for i in range(nints):
            for j in range(nints):
                for k in range(nints):
                    if F3[i,j,k] != 0:
                        print(i,j,k,F3[i,j,k])
                    for l in range(nints):
                        if F4[i,j,k,l] != 0:
                            print(i,j,k,l,F4[i,j,k,l])
                        if order > 4:
                            for m in range(nints):
                                if F5[i,j,k,l,m] != 0:
                                    print(i,j,k,l,m,F5[i,j,k,l,m])
                                for n in range(nints):
                                    if F6[i,j,k,l,m,n] != 0:
                                        print(i,j,k,l,m,n,F6[i,j,k,l,m,n])
        print()
    if False:
        print("Taylor expansion at single dissociation (x=xe, y=oo, t=te):")
        for i in range(7):
            print(method, i, 'x', sym.limit(sym.diff(G,X_S[1],i).subs(X_S[1],X_0[1]).subs(X_S[2],X_0[2]),X_S[0],sym.oo)) 
        for i in range(7):
            print(method, i, 't', sym.limit(sym.diff(G,X_S[2],i).subs(X_S[1],X_0[1]).subs(X_S[2],X_0[2]),X_S[0],sym.oo)) 
        print("Taylor expansion at double dissociation (x=oo, y=oo, t=te):")
        for i in range(7):
            print(method, i, 't', sym.limit(sym.limit(sym.diff(G,X_S[2],i).subs(X_S[2],X_0[2]),X_S[0],sym.oo),X_S[1],sym.oo)) 
        print()
    return F 

def pypes(filename,F,Z,X,X_i,A,notes,order=6):
    natoms, bonds, angs, oops, nors, tors = read.defconn(A)
    nbonds = len(bonds)
    nangs = len(angs) 
    noops = len(oops)
    ntors = len(tors)
    nints = nbonds + nangs + noops + ntors  
    file = open( filename, 'w' )
    prologue = """from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
import math
import sympy
import state
from useful import assign_connectivity
from useful import change_connectivity
from useful import empty_connectivity
from useful import assign_internal_params
from useful import retrieve_coords_int_eq
from force_field import force_field_class
def initialise_force_field():
  ff = force_field_class.Force_field()
"""
    epilogue = """  else:
    print ('Fatal error: PESN = '+state.pesn+' does not exist')
    sys.exit()
  return ff
"""
    file.write(prologue)
    for p in range(nbonds):
        file.write("  assign_connectivity( state.conn, int_type='BL', i_int=" + str(p+1) + ", int_conn=[" + str(bonds[p][0]+1) + "," + str(bonds[p][1]+1) + "] ) \n" )
    for r in range(nangs):
        file.write("  assign_connectivity( state.conn, int_type='BA', i_int=" + str(nbonds+r+1) + ", int_conn=[" + str(angs[r][0]+1) + "," + str(angs[r][1]+1) + "," + str(angs[r][2]+1) + "]) \n" )
    for s in range(noops):
        file.write("  assign_connectivity( state.conn, int_type='OOP1', i_int=" + str(nbonds+nangs+s+1) + ", int_conn=[" + str(oops[s][0]+1) + "," + str(oops[s][1]+1) + "," + str(oops[s][2]+1) + "," + str(oops[s][3]+1) + "]) \n" )
    for t in range(ntors):
        file.write("  assign_connectivity( state.conn, int_type='DA', i_int=" + str(nbonds+nangs+noops+t+1) + ", int_conn=[" + str(tors[t][0]+1) + "," + str(tors[t][1]+1) + "," + str(tors[t][2]+1) + "," + str(tors[t][3]+1) + "]) \n" )
    file.write("  state.Nint = " + str(nints) + "\n")
    file.write("  state.atm_type = " + str(Z).upper() + "\n")
    file.write("  if state.pesn == '1':\n")
    file.write("    state.ff_comment.append(' Internal coordinates automatically generated from PSI4 input:')\n")
    file.write("    state.ff_comment.append('')\n")
    file.write("    state.ff_comment.append('   " + filename + "')\n")
    for note in notes:
        file.write("    state.ff_comment.append('   " + note + "')\n")
    file.write("    state.ff_comment.append('')\n")
    if units[0] * units[1] != 1:
        #file.write("        ff.units = ['aJ', 'angst', 'rad']\n")
        print("UNITS!")
        exit()
    file.write("    ff.equilibrium_coords_cart = " + np.array2string(X,separator=", ") + "\n")
    for i in range(nints):
        file.write("    r" + str(i+1) + " = " + str(X_i[i]) + "\n") 
    for i in range(nints):
        file.write("    assign_internal_params( state.conn, state.internal_params, i_int=" + str(i+1) + ", params=[r" + str(i+1) + "] )\n")
    file.write("    ff.equilibrium_coords_int = retrieve_coords_int_eq(state.conn,state.internal_params,ff.units)\n")
    file.write("    ff.ff = [\n")
    for i in range(nints):
        for j in range(i,nints):
            file.write("          [" + str(i) + "," + str(j) + ", " + str(F[2][i,j]) + " ],\n")
            if order > 2:
                for k in range(j,nints):
                    if F[3][i,j,k] != 0:
                        file.write("          [" + str(i) + "," + str(j) + "," + str(k) + ", " + str(F[3][i,j,k]) + " ],\n")
                    for l in range(k,nints):
                        if F[4][i,j,k,l] != 0:
                            file.write("          [" + str(i) + "," + str(j) + "," + str(k) + "," + str(l) + ", " + str(F[4][i,j,k,l]) + " ],\n")
                        if order > 4:
                            for m in range(l,nints):
                                if F[5][i,j,k,l,m] != 0:
                                    file.write("          [" + str(i) + "," + str(j) + "," + str(k) + "," + str(l) + "," + str(m) + ", " + str(F[5][i,j,k,l,m]) + " ],\n")
                                for n in range(m,nints):
                                    if F[6][i,j,k,l,m,n] != 0:
                                        file.write("          [" + str(i) + "," + str(j) + "," + str(k) + "," + str(l) + "," + str(m) + "," + str(n) + ", " + str(F[6][i,j,k,l,m,n]) + " ],\n")
    file.write("]\n")
    file.write(epilogue)
    file.close()

