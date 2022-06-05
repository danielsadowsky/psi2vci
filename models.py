import math
import sympy as sym
import numpy as np
from scipy import optimize

def eval_eq(function,variables,values):
    n = len(variables)
    V = function
    for i in range(n):
        V = V.subs(variables[i],values[i])
    return V

def res_hess(x,y,data,K=False):
    H_i, X_S, X_i, U = data
    if type(K) == float:
        return eval_eq( sym.diff( sym.diff( U, X_S[x] ), X_S[y] ), X_S, X_i ) - K 
    else:
        return eval_eq( sym.diff( sym.diff( U, X_S[x] ), X_S[y] ), X_S, X_i ) - H_i[x,y]

def res_grad(x,data,J=False):
    H_i, X_S, X_i, U = data
    if type(J) == float:
        return eval_eq( sym.diff( U, X_S[x] ), X_S, X_i  ) - J
    else:
        return eval_eq( sym.diff( U, X_S[x] ), X_S, X_i ) 

def bicoord_harmonic(G,H_i,X_S,X_i,indices,U_0=0.0):
    p, q, r = indices
    G += H_i[p,q] * ( X_S[p] - X_i[p] ) * ( X_S[q] - X_i[q] ) 
    G += 0.5 * H_i[r,r] * ( X_S[r] - X_i[r] )**2 - U_0
    if abs( X_i[r] - math.pi ) > 1e-4:
        G += H_i[p,r] * ( X_S[p] - X_i[p] ) * ( X_S[r] - X_i[r] )
        G += H_i[q,r] * ( X_S[q] - X_i[q] ) * ( X_S[r] - X_i[r] )
    return G

def bicoord_linear(G,H_i,X_S,X_i,indices,R_S,n):
    p, q, r = indices  
    n_f = 1
    inverse_p = X_S[p]**(-n_f)
    inverse_q = X_S[q]**(-n_f)
    inverse_prefactor = inverse_p * inverse_q 
    def model(parameters,optimization=True):
        U = parameters[0] * inverse_prefactor
        U += parameters[1] * inverse_p * inverse_prefactor
        U += parameters[2] * inverse_q * inverse_prefactor
        U += parameters[3] * inverse_p * inverse_q * inverse_prefactor
        U += parameters[4] * inverse_p**2 * inverse_prefactor
        U += parameters[5] * inverse_q**2 * inverse_prefactor
        U += parameters[6] * R_S[0]**(-n)
        if optimization == True:
            data = H_i, X_S, X_i, U
            residuals = []
            residuals.append( res_hess( p, q, data ) )
            residuals.append( res_hess( r, r, data ) )
            residuals.append( res_hess( p, p, data, 0.0 ) ) 
            residuals.append( res_hess( q, q, data, 0.0 ) ) 
            residuals.append( res_grad( p, data ) ) 
            residuals.append( res_grad( q, data ) ) 
            residuals.append( eval_eq( U, X_S, X_i ) ) 
            if False:
                print(parameters)
                print(residuals)
            return residuals 
        elif optimization == "U":
            return U
    guess = [ 0.0 ] * 6  + [ 10.0 ]
    sol = optimize.fsolve( model, guess, factor=0.1, xtol=1.0e-8 )
    if True:
        U = model(sol,"U")
        print("FORCE FIELD PARAMETERS:")
        print(sol)
        print()
    G += model(sol,"U")
    return G

def bicoord_bent(G,H_i,X_S,X_i,indices,R_S,n):
    p, q, r = indices  
    n_f = 1
    inverse_p = X_S[p]**(-n_f)
    inverse_q = X_S[q]**(-n_f)
    inverse_prefactor = inverse_p * inverse_q 
    def model(parameters,optimization=True):
        U = parameters[0] * inverse_prefactor
        U += parameters[1] * inverse_p * inverse_prefactor
        U += parameters[2] * inverse_q * inverse_prefactor
        U += parameters[3] * inverse_p * inverse_q * inverse_prefactor
        U += parameters[4] * inverse_p**2 * inverse_prefactor
        U += parameters[5] * inverse_q**2 * inverse_prefactor
        U += parameters[6] * R_S[0]**(-n)
        U += ( parameters[7] + parameters[8] * X_S[p] + parameters[9] * X_S[q] ) * R_S[1]**(-n)
        if optimization == True:
            data = H_i, X_S, X_i, U
            residuals = []
            residuals.append( res_hess( p, q, data ) )
            residuals.append( res_hess( p, r, data ) )
            residuals.append( res_hess( q, r, data ) )
            residuals.append( res_hess( r, r, data ) )
            residuals.append( res_hess( p, p, data, 0.0 ) ) 
            residuals.append( res_hess( q, q, data, 0.0 ) ) 
            residuals.append( res_grad( p, data ) ) 
            residuals.append( res_grad( q, data ) ) 
            residuals.append( res_grad( r, data ) ) 
            residuals.append( eval_eq( U, X_S, X_i ) ) 
            if False:
                print(parameters)
                print(residuals)
            return residuals 
        elif optimization == "U":
            return U
    guess = [ 0.0 ] * 6  + [ 10.0, 10.0, 0.0, 0.0 ]
    sol = optimize.fsolve( model, guess, factor=0.1, xtol=1.0e-8 )
    if True:
        print("FORCE FIELD PARAMETERS:")
        print(sol)
        print()
    G += model(sol,"U")
    return G

def tricoord_harmonic(G,H_i,X_S,X_i,indices): 
    p, q, r, t_p, t_q, t_r, s = indices  
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
    G = bicoord_harmonic(G, H_i, X_S, X_i, ( p, q, t_r ) )
    G = bicoord_harmonic(G, H_i, X_S, X_i, ( q, r, t_p ) )
    G = bicoord_harmonic(G, H_i, X_S, X_i, ( p, r, t_q ) )
    return G

def tricoord_symmetric_planar(G,H_i,X_S,X_i,indices,R_S,n):
    p, q, r, t_p, t_q, t_r, s = indices  
    n_f = 1
    inverse_p = X_S[p]**(-n_f)
    inverse_q = X_S[q]**(-n_f)
    inverse_r = X_S[r]**(-n_f)
    inverse_prefactor = inverse_p * inverse_q * inverse_r
    def model(parameters,optimization=True):
        U = parameters[0] * inverse_prefactor
        U += parameters[1] * inverse_p * inverse_prefactor
        U += parameters[1] * inverse_q * inverse_prefactor
        U += parameters[1] * inverse_r * inverse_prefactor
        U += parameters[2] * inverse_p * inverse_q * inverse_prefactor
        U += parameters[2] * inverse_p * inverse_r * inverse_prefactor
        U += parameters[2] * inverse_q * inverse_r * inverse_prefactor
        U += parameters[3] * inverse_p**2 * inverse_prefactor
        U += parameters[3] * inverse_q**2 * inverse_prefactor
        U += parameters[3] * inverse_r**2 * inverse_prefactor
        if True:
            U += parameters[4] * inverse_prefactor * X_S[s]**2
        else:
            U += parameters[4] * ( X_S[0] + X_S[2] + X_S[4] )**(-n) 
        U += parameters[5] * R_S[0]**(-n)
        U += parameters[5] * R_S[2]**(-n)
        U += parameters[5] * R_S[4]**(-n)
        if True:
            U += parameters[6] * ( X_S[p] + X_S[q] + R_S[0] )**(-n)
            U += parameters[6] * ( X_S[p] + X_S[r] + R_S[2] )**(-n)
            U += parameters[6] * ( X_S[q] + X_S[r] + R_S[4] )**(-n)
        if True:
            U += parameters[7] * ( X_S[r] + R_S[0] )**(-n)
            U += parameters[7] * ( X_S[q] + R_S[2] )**(-n)
            U += parameters[7] * ( X_S[p] + R_S[4] )**(-n)
        if True:
            U += parameters[8] * (R_S[0] + R_S[2])**(-n)
            U += parameters[8] * (R_S[2] + R_S[4])**(-n)
            U += parameters[8] * (R_S[4] + R_S[0])**(-n)
        if optimization == True:
            data = H_i, X_S, X_i, U
            residuals = []
            residuals.append( res_hess( p, q, data ) )
            residuals.append( res_hess( t_p, t_p, data ) )
            residuals.append( res_hess( t_p, t_q, data ) )
            residuals.append( res_hess( t_p, p, data ) )
            residuals.append( res_hess( t_p, r, data ) )
            residuals.append( res_hess( s, s, data ) )
            residuals.append( res_hess( p, p, data, 0.0 ) ) 
            residuals.append( res_grad( p, data ) ) 
            residuals.append( eval_eq( U, X_S, X_i ) ) 
            #residuals.append( res_hess( q, q, data, 0.0 ) ) 
            #residuals.append( res_hess( p, r, data ) )
            #residuals.append( res_hess( q, r, data ) )
            #residuals.append( res_hess( t_q, t_q, data ) )
            #residuals.append( res_hess( t_r, t_r, data ) )
            #residuals.append( res_hess( t_p, t_q, data ) )
            #residuals.append( res_hess( t_r, r, data ) )
            #residuals.append( res_hess( t_r, p, data ) )
            #residuals.append( res_hess( t_q, q, data ) )
            #residuals.append( res_hess( t_r, r, data ) )
            #residuals.append( res_hess( r, r, data, 0.0 ) ) 
            #residuals.append( res_grad( q, data ) ) 
            #residuals.append( res_grad( r, data ) ) 
            if False:
                print(parameters)
            return residuals 
        elif optimization == "U":
            return U
    guess = [ 0.0 ] * 4  + [ H_i[s,s] / 2 * X_i[p] * X_i[q] * X_i[r] ] + [ 10.0, 10.0, 10.0, 10.0 ]  
    sol = optimize.fsolve( model, guess, factor=0.1, xtol=1.0e-8 )
    if True:
        print("FORCE FIELD PARAMETERS:")
        print(sol)
        print()
    G += model(sol,"U")
    return G

def tetracoord_harmonic(G,H_i,X_S,X_i,indices,angs):
    for j in range(6):
        p_a = angs[ j ][3]
        q_a = angs[ j ][4]
        G = bicoord_harmonic(G, H_i, X_S, X_i, ( p_a, q_a, indices[j] ) )
        for k in range(j+1,6):
            G += H_i[ indices[j], indices[k] ] * ( X_S[ indices[j] ] - X_i[ indices[j] ] ) * ( X_S[ indices[k] ] - X_i[ indices[k] ] )
            p_b = angs[ k ][3]
            q_b = angs[ k ][4]
            opposite_bonds = [ p_b, q_b ]
            if not ( p_a in opposite_bonds ) and not ( q_a in opposite_bonds):
                G += H_i[ q_b, indices[j] ] * ( X_S[ indices[j] ] - X_i[ indices[j] ] ) * ( X_S[q_b] - X_i[q_b] )
                G += H_i[ q_a, indices[k] ] * ( X_S[ indices[k] ] - X_i[ indices[k] ] ) * ( X_S[q_a] - X_i[q_a] )
                G += H_i[ p_b, indices[j] ] * ( X_S[ indices[j] ] - X_i[ indices[j] ] ) * ( X_S[p_b] - X_i[p_b] )
                G += H_i[ p_a, indices[k] ] * ( X_S[ indices[k] ] - X_i[ indices[k] ] ) * ( X_S[p_a] - X_i[p_a] )
    return G

def tetracoord_tetrahedral(G,H_i,X_S,X_i,indices,angs,n):
    opposites = []
    R_S = []
    for j in range(6):
        p_a = angs[j][3]
        q_a = angs[j][4]
        r = indices[j]
        R_S.append( sym.sqrt( X_S[p_a]**2 + X_S[q_a]**2 - 2 * X_S[p_a] * X_S[q_a] * sym.cos( X_S[r] ) ) )
        R_S.append( sym.sqrt( X_S[p_a]**2 + X_S[q_a]**2 - 2 * X_S[p_a] * X_S[q_a] * sym.cos( math.pi - 0.5 * X_S[r] ) ) )
        for k in range(j+1,6):
            p_b = angs[k][3]
            q_b = angs[k][4]
            opposite_bonds = [ p_b, q_b ]
            if not ( p_a in opposite_bonds ) and not ( q_a in opposite_bonds):
                p, q, r, s = p_a, q_a, p_b, q_b
                opposites.append( (j, k) )
    a, b, c, d, e, f = tuple(indices)
    n_f = 1
    inverse_p = X_S[p]**(-n_f)
    inverse_q = X_S[q]**(-n_f)
    inverse_r = X_S[r]**(-n_f)
    inverse_s = X_S[s]**(-n_f)
    inverse_prefactor = inverse_p * inverse_q * inverse_r * inverse_s
    def model(parameters,optimization=True):
        U = parameters[0] * inverse_prefactor
        U += parameters[1] * inverse_p * inverse_prefactor
        U += parameters[1] * inverse_q * inverse_prefactor
        U += parameters[1] * inverse_s * inverse_prefactor
        U += parameters[1] * inverse_r * inverse_prefactor
        U += parameters[2] * inverse_p * inverse_q * inverse_prefactor
        U += parameters[2] * inverse_p * inverse_r * inverse_prefactor
        U += parameters[2] * inverse_p * inverse_s * inverse_prefactor
        U += parameters[2] * inverse_q * inverse_r * inverse_prefactor
        U += parameters[2] * inverse_q * inverse_s * inverse_prefactor
        U += parameters[2] * inverse_r * inverse_s * inverse_prefactor
        U += parameters[3] * inverse_p**2 * inverse_prefactor
        U += parameters[3] * inverse_q**2 * inverse_prefactor
        U += parameters[3] * inverse_r**2 * inverse_prefactor
        U += parameters[3] * inverse_s**2 * inverse_prefactor
        # VALID
        U += parameters[4] * R_S[0]**(-n)
        U += parameters[5] * R_S[1]**(-n)
        U += parameters[4] * R_S[2]**(-n)
        U += parameters[5] * R_S[3]**(-n)
        U += parameters[4] * R_S[4]**(-n)
        U += parameters[5] * R_S[5]**(-n)
        U += parameters[4] * R_S[6]**(-n)
        U += parameters[5] * R_S[7]**(-n)
        U += parameters[4] * R_S[8]**(-n)
        U += parameters[5] * R_S[9]**(-n)
        U += parameters[4] * R_S[10]**(-n)
        U += parameters[5] * R_S[11]**(-n)
        if optimization == True:
            data = H_i, X_S, X_i, U
            residuals = []
            residuals.append( res_hess( a, a, data ) )
            #residuals.append( res_hess( b, b, data ) )
            #residuals.append( res_hess( c, c, data ) )
            #residuals.append( res_hess( d, d, data ) )
            #residuals.append( res_hess( e, e, data ) )
            #residuals.append( res_hess( f, f, data ) )
            residuals.append( res_hess( a, p, data ) )
            residuals.append( res_hess( p, q, data ) )
            #residuals.append( res_hess( p, r, data ) )
            #residuals.append( res_hess( p, s, data ) )
            #residuals.append( res_hess( q, r, data ) )
            #residuals.append( res_hess( q, s, data ) )
            #residuals.append( res_hess( r, s, data ) )
            residuals.append( res_hess( p, p, data, 0.0 ) ) 
            #residuals.append( res_hess( q, q, data, 0.0 ) ) 
            #residuals.append( res_hess( r, r, data, 0.0 ) ) 
            #residuals.append( res_hess( s, s, data, 0.0 ) ) 
            residuals.append( res_grad( p, data ) ) 
            #residuals.append( res_grad( q, data ) ) 
            #residuals.append( res_grad( r, data ) ) 
            #residuals.append( res_grad( s, data ) ) 
            residuals.append( eval_eq( U, X_S, X_i ) ) 
            if True:
                print(parameters)
                #print(residuals)
            return residuals 
        elif optimization == "U":
            return U
    guess = [ 0.0 ] * 4 + [ 6.0, 6.0 ] 
    sol = optimize.fsolve( model, guess, factor=0.1, xtol=1.0e-8 )
    if True:
        print("FORCE FIELD PARAMETERS:")
        print(sol)
        print()
    G += model(sol,"U")
    return G

def hexacoord_harmonic(G,H_i,X_S,X_i,indices,angs):
    bonds = []
    for j in range(15):
        p = angs[ j ][3]
        q = angs[ j ][4]
        if not p in bonds:
            bonds.append(p)
        if not q in bonds:
            bonds.append(q)
    print(bonds)   
    for j in range(15):
        p = angs[ j ][3]
        q = angs[ j ][4]
        G = bicoord_harmonic(G, H_i, X_S, X_i, ( p, q, indices[j] ) )
        for k in range(j+1,15):
            G += H_i[ indices[j], indices[k] ] * ( X_S[ indices[j] ] - X_i[ indices[j] ] ) * ( X_S[ indices[k] ] - X_i[ indices[k] ] )
        for l in bonds:
            if l != p and l != q:
                G += H_i[ l, indices[j] ] * ( X_S[ indices[j] ] - X_i[ indices[j] ] ) * ( X_S[l] - X_i[l] )
    return G


