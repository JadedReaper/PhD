import numpy as np
import numpy.linalg as la
from qutip import *
from itertools import product
import matplotlib.pyplot as plt

def eigdecomp(V, tol=1e-12):
    eigvals, eigvecs = la.eig(V)
    W = np.zeros_like(V, dtype=V.dtype)

    for i, val in enumerate(eigvals):
        if val> tol:
            vec = eigvecs[:, i]
            W += np.sqrt(val) * np.outer(vec, vec.conj())

    return W, W.conj().T

def build_gamma(W_r, W_l, Axij):
    """
    Construct Gamma^s = Vh @ W_r^{-1} @ A_s @ W_l^{-1} @ U
    for every physical symbol s and return array of shape (d, D, D).
    """
    C = W_l @ W_r
    [U,S,Vh] = la.svd(C,full_matrices=False)
    S_mat = np.diag(S) # singular value = sqrt lambda
    d = Axij.shape[0]
    Gammas = []
    for s in range(d):
        Gamma_s = Vh @ la.inv(W_r) @ Axij[s] @ la.inv(W_l) @ U
        Gammas.append(Gamma_s)
    return np.array(Gammas), S_mat

def QPC(p):


    ################# define process
    B0 = np.array([[np.sqrt(p), np.sqrt(1-p)], [0, 0]])
    B1 = np.array([[0, 0], [np.sqrt(1-p), np.sqrt(p)]])

    Bx = np.array([B0, B1])
    D = B0.shape[0]

    ################# transfer matrix and leading eigval and vect
    E = np.einsum('xij,xkl->ikjl', Bx, Bx.conj()).reshape(D**2, D**2)

    vals_L, vecs_L = la.eig(E.T)
    idx_L = np.argmax(np.abs(vals_L))
    val_L = vals_L[idx_L]
    v_L = vecs_L[:, idx_L]

    vals_R, vecs_R = la.eig(E)
    idx_R = np.argmax(np.abs(vals_R))
    val_R = vals_R[idx_R]
    v_R = vecs_R[:, idx_R]
    #print("Leading eigenvalue:", val_L, val_R)
    
    ##### check how to normalise



    ##################### RIGHT canonical form

    # Matricise eigvects
    V_R = v_R.reshape(D, D)
    V_L = v_L.reshape(D, D)

 
    nV_L = V_L / np.trace(V_L)  # such that prob sum to 1
    overlap = np.trace(nV_L.conj().T @ V_R)
    nV_R = V_R / overlap
    #print(nV_R, nV_L)

    # HOW TO NORMALISE??????????????????????????????????????????

    # compare to by hand check using B0, B1 as kraus operators
    #sig = np.array([[1/2, np.sqrt(p*(1-p))], [np.sqrt(p*(1-p)), 1/2]])
    #print(np.allclose(V_L/np.trace(V_L), sig/np.trace(sig), atol=1e-12))


    W_L, W_L_dag = eigdecomp(nV_L)
    W_R, W_R_dag = eigdecomp(nV_R)

    Gammax, S_mat = build_gamma(W_R, W_L, Bx)

    Kx_R = Gammax @ S_mat

    #check right canonical form]
    check = sum(K @ K.conj().T for K in Kx_R)

    if np.allclose(check, np.eye(D), atol=1e-12) == False:
        print("NOT Right canonical form")

    check2 = sum(B @ B.conj().T for B in Bx)

    if np.allclose(check2, np.eye(D), atol=1e-12) == False:
        print("NOT Right canonical form")



    return nV_L, nV_R, Kx_R


def entropy (p):
    VL, VR, Kx = QPC(p)
    #print('Cq before can', entropy_vn(Qobj(VL), base=2))
    D = Kx.shape[0]
    E = np.einsum('xij,xkl->ikjl', Kx, Kx.conj()).reshape(D**2, D**2)

    vals_L, vecs_L = la.eig(E.T)
    idx_L = np.argmax(np.abs(vals_L))
    val_L = vals_L[idx_L]
    v_L = vecs_L[:, idx_L]

    vals_R, vecs_R = la.eig(E)
    idx_R = np.argmax(np.abs(vals_R))
    val_R = vals_R[idx_R]
    v_R = vecs_R[:, idx_R]
    #print('val', val_L, val_R)
    # Matricise eigvects
    V_R = v_R.reshape(D, D)
    V_L = v_L.reshape(D, D)

    nV_L = V_L / np.trace(V_L)  # such that prob sum to 1
    overlap = np.trace(nV_L.conj().T @ V_R)
    nV_R = V_R / overlap
    #print(nV_R, nV_L)

    return entropy_vn(Qobj(nV_L), base=2)


# print('p=0.1', entropy(0.1))
# print('p=0.6', entropy(0.6))
# print('p=0.8', entropy(0.8))

# matches values given in QS paper!

############# Block L stuff

def block_entropy_brute_force(p):

    VL, VR, KxR = QPC(p)

    B0 = np.array([[np.sqrt(p), np.sqrt(1-p)], [0, 0]])
    B1 = np.array([[0, 0], [np.sqrt(1-p), np.sqrt(p)]])

    E00 = np.einsum('ij,kl->ikjl',
                   B0, B0.conj()).reshape(4,4)

    E11 = np.einsum('ij,kl->ikjl',
                   B1, B1.conj()).reshape(4,4)
    
    E01 = np.einsum('ij,kl->ikjl',
                   B0, B1.conj()).reshape(4,4)

    E10 = np.einsum('ij,kl->ikjl',
                   B1, B0.conj()).reshape(4,4)



    id = VR.reshape(4)
    sigma = VL.reshape(4)


    # L=2
    P00 = sigma @ E00 @ id
    P11 = sigma @ E11 @ id
    P01 = sigma @ E01 @ id
    P10 = sigma @ E10 @ id

    rho1 = np.array([[P00, P01], [P10, P11]], dtype=complex)
    S1 = entropy_vn(Qobj(rho1), base=2)

    P00_00 = sigma @ E00 @ E00 @ id
    P00_11 = sigma @ E00 @ E11 @ id
    P11_00 = sigma @ E11 @ E00 @ id
    P11_11 = sigma @ E11 @ E11 @ id
    P01_00 = sigma @ E01 @ E00 @ id
    P01_11 = sigma @ E01 @ E11 @ id
    P10_00 = sigma @ E10 @ E00 @ id
    P10_11 = sigma @ E10 @ E11 @ id
    P00_01 = sigma @ E00 @ E01 @ id
    P00_10 = sigma @ E00 @ E10 @ id
    P11_01 = sigma @ E11 @ E01 @ id
    P11_10 = sigma @ E11 @ E10 @ id
    P01_01 = sigma @ E01 @ E01 @ id
    P01_10 = sigma @ E01 @ E10 @ id
    P10_01 = sigma @ E10 @ E01 @ id
    P10_10 = sigma @ E10 @ E10 @ id
    
    rho2 = np.array([
    [P00_00, P00_01, P01_00, P01_01],
    [P00_10, P00_11, P01_10, P01_11],
    [P10_00, P10_01, P11_00, P11_01],
    [P10_10, P10_11, P11_10, P11_11]], dtype=complex)

    S2 = entropy_vn(Qobj(rho2), base=2)

    return S1, S2

def block_entropy_expensive(p, L):

    # uses B0 amd B1 straight
    # if i wanted to use right canonical form, would get Kx and compute Vl and Vr of transfer matrix built of Kx
    # when is this neccesary?????????????

    VL, VR, KxR = QPC(p)

    B0 = np.array([[np.sqrt(p), np.sqrt(1-p)], [0, 0]])
    B1 = np.array([[0, 0], [np.sqrt(1-p), np.sqrt(p)]])

    B = [B0, B1]
   

    # E[x,y]
    E = np.empty((2, 2, 4, 4), dtype=complex)

    for x in range(2):
        for y in range(2):
            E[x, y] = np.einsum(
                'ij,kl->ikjl',
                B[x], B[y].conj()
            ).reshape(4, 4)

    id_vec = VR.reshape(4)
    sigma = VL.reshape(4)

    # All binary words of length L
    words = list(product([0, 1], repeat=L))

    # rho_L has dimension 2^L x 2^L
    rhoL = np.zeros((2**L, 2**L), dtype=complex)

    for i, xword in enumerate(words):
        for j, yword in enumerate(words):

            v = id_vec

            # Gives E[x1,y1] E[x2,y2] ... E[xL,yL] @ id_vec
            # because we're multiplying from the right.
            for n in reversed(range(L)):
                v = E[xword[n], yword[n]] @ v

            rhoL[i, j] = sigma @ v

    S_L = entropy_vn(Qobj(rhoL), base=2)

    return S_L

def plot_block_entropy(L, fn):
    L = range(1, L)
    S1 = []
    Cq1 = []
    for l in L:
        S1.append(fn(0.1, l))
        Cq1.append(2*entropy(0.1))

    S2 = []
    Cq2 = []
    for l in L:
        S2.append(fn(0.6, l))
        Cq2.append(2*entropy(0.6))

    S3 = []
    Cq3 = []
    for l in L:
        S3.append(fn(0.8, l))
        Cq3.append(2*entropy(0.8))


    plt.figure(figsize=(6,4))
    plt.plot(L, S1, color='blue', marker='o', lw=2, linestyle='None', label='p=0.1')
    plt.plot(L, Cq1, color='black', linestyle='--', lw=2)
    plt.plot(L, S2, color='red', marker='o', lw=2, linestyle='None', label='p=0.6')
    plt.plot(L, Cq2, color='black', linestyle='--', lw=2)
    plt.plot(L, S3, color='green', marker='o', lw=2, linestyle='None', label='p=0.8')
    plt.plot(L, Cq3, color='black', linestyle='--', lw=2)
    plt.xlabel(r'$L$')
    plt.xscale('log')
    plt.ylabel(r'$S(\rho)$')
    plt.legend()
    plt.title('QPC')
    plt.grid(True)
    plt.show()

    return None

#plot_block_entropy(0.6, 3)



# Isometry!

###############################
# B0 = np.array([[np.sqrt(p), np.sqrt(1-p)], [0, 0]])
# B1 = np.array([[0, 0], [np.sqrt(1-p), np.sqrt(p)]])

# Bx = np.array([B0, B1])  # xij

# VL, VR, Kx = QPC(p)
# sigma = VL

# p, U = np.linalg.eigh(sigma)

# psi = np.zeros(4, dtype=complex)

# for j in range(2):
#     ket_j = U[:, j]
#     psi += np.sqrt(p[j]) * np.kron(ket_j, ket_j)


# psi = psi.reshape(2,2)
# L1 = np.einsum('ij, xjk -> xik', psi, Bx)
# P1 = np.einsum('xij, xkl -> ijkl', L1, L1).reshape(4,4)

# L2 = np.einsum('xij, yjk -> xyik', L1, Bx)
# P2 = np.einsum('xij, xkl -> ijkl', L2, L2).reshape(4,4)




def contract_E(E, L):
    state = E.copy()
    for _ in range(1, L):
        state = np.einsum('', state, E)
    return state

def block_entropy_cheap(p, L):

    B0 = np.array([[np.sqrt(p), np.sqrt(1-p)], [0, 0]])
    B1 = np.array([[0, 0], [np.sqrt(1-p), np.sqrt(p)]])

    Bx = np.array([B0, B1])  # xij

    E = np.einsum('', Bx, Bx.conj())


    sigma, VR, Kx = QPC(p)

    p, U = np.linalg.eigh(sigma)

    psi = np.zeros(4, dtype=complex)

    for j in range(2):
        ket_j = U[:, j]
        psi += np.sqrt(p[j]) * np.kron(ket_j, ket_j)

    psi = psi.reshape(2,2)

    PSI = np.einsum('', psi, psi)

    Es = contract_E(E, L)
    state = np.einsum('', PSI, Es)
    

    S = entropy_vn(Qobj(state.reshape(4,4)), 2)

    return S

print(block_entropy_cheap(0.1, 2))
print(block_entropy_cheap(0.1, 7))
    

    


# plt.figure(figsize=(6,4))
# plt.plot(L, Ss, color='purple', marker='o', lw=2, linestyle='None', label='p=0.1')
# plt.xlabel(r'$L$')
# plt.xscale('log')
# plt.ylabel(r'$S(\rho)$')
# plt.legend()
# plt.title('QPC')
# plt.grid(True)
# plt.show()
