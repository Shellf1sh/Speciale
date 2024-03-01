import numpy as np

from scipy import sparse

def kron_L(L, L_res, I_dim): #This only works for square matrices, but it can be generalised
    L = sparse.lil_matrix(L)
    L_dim = int(L.shape[0]/L_res)
    I = sparse.eye(int(I_dim), format = "lil")
    new = sparse.lil_matrix((L.shape[0]*I_dim, L.shape[0]*I_dim))

    for m in range(0, L_dim*I_dim):
        for n in range(0, L_dim*I_dim):

            sub_matrice = L[(m//I_dim)*L_res: (m//I_dim+1)*L_res,(n//I_dim)*L_res: (n//I_dim+1)*L_res]*I[m%I_dim,n%I_dim]

            new[L_res*m:L_res*(m+1), L_res*n:L_res*(n+1)] = sub_matrice
    return new

def kron_R(I_dim, R, R_res):
    R = sparse.lil_matrix(R)
    R_dim = int(R.shape[0]/R_res)
    I = sparse.eye(int(I_dim), format = "lil")
    new = sparse.lil_matrix((R.shape[0]*I_dim, R.shape[0]*I_dim))

    for m in range(0, R_dim*I_dim):
        for n in range(0, R_dim*I_dim):

            sub_matrice = I[m//I_dim, n//I_dim] * R[(m%I_dim)*R_res:(m%I_dim+1)*R_res, (n%I_dim)*R_res:(n%I_dim+1)*R_res]

            new[R_res*m:R_res*(m+1), R_res*n:R_res*(n+1)] = sub_matrice
    return new

def kron_sum(L, L_res, R, R_res):
    L_dim_kron_sum = int(L.shape[0]/L_res)
    R_dim_kron_sum = int(R.shape[0]/R_res)

    total = kron_L(L, L_res, R_dim_kron_sum) + kron_R(L_dim_kron_sum, R, R_res)

    return total
