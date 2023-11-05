import numpy as np


def two_block_solution(A):
    b = -A[0, 0] - A[1, 1]
    c = - A[0, 1] * A[1, 0] + A[0, 0] * A[1, 1]

    det = b**2 - 4 * c
    ext = - b / 2
    wing = np.sqrt(abs(det)) / 2


    if det >= 0:
        return [(ext - wing, 0), (ext + wing, 0)]
    else:
        return [(ext, -wing), (ext, wing)]
    

def householder(v):

    sgn = -1 if v[0] > 0 else 1
    beta = sgn * np.linalg.norm(v)
    w = v.copy()
    w[0] = w[0] - beta
    w = w / (np.linalg.norm(w)  + 1e-20)

    return np.identity(len(w)) - 2 *np.outer(w, w) 


def sim_qr(T, n):
    G = givenc(T[0: 2, 0]) 
    for i in range(n - 1):   
        m = min(i + 3, n)
        k = max(0, i - 1)
        T[i: i+2, k: m] = G @ T[i: i+2, k: m]    
        if i < n-2:
            future_G = givenc(T[i + 1: i + 3, i + 1]) 
        
        T[k:m, i: i + 2] = T[k:m, i: i + 2] @ G.T
        G = future_G


def hessenberg(A):
    n = len(A)
    R = A.copy()

    for i in range(n - 2):
        norm = np.linalg.norm(R[i + 2:, i])

        if norm > 0:
            G = householder(R[i + 1:, i])
            R[i + 1:, i:] = G @ R[i + 1:, i:]
            R[:, i + 1:] = R[:, i + 1:] @ G.T

    return R


def givenc(x):
    r = np.sqrt(x[0]**2 + x[1]**2 + 1e-20)
    cos = x[0] / r
    sin = x[1] / r

    return np.array([[cos, sin], [-sin, cos]])


def top_qr(R, n):
    G = givenc(R[:2, 0])
    for i in range(n - 1):
        
        R[i:i+2, :n] = G @ R[i:i+2, :n]
        if i < n - 2:
            future_G = givenc(R[i + 1: i+3, i + 1])
        R[:n, i: i+ 2] = R[:n, i: i+ 2] @ G.T

        G = future_G


def eigenvalues(A, eps=1e-12, method="simple", simmetric=False, info_mode=True):

    H = hessenberg(A)
    n = len(A)
    counter = 0
    eigenvalues = []
    sigma = 0

    while n > 0:
        if n == 1:
            eigenvalues.append(H[0, 0])
            n -= 1
        elif n == 2:
            z1, z2 = two_block_solution(H[0:2, 0:2])

            eigenvalues.append(z1[0] + 1j * z1[1])
            eigenvalues.append(z2[0] + 1j * z2[1])

            n -= 2
        else:
            
            while abs(H[n - 1, n - 2]) >= eps and abs(H[n - 2, n - 3]) >= eps:
                if method == "simple":
                    sigma = H[n - 1, n - 1]
                if method == "wilkinson":
                    A = H[n - 2:n, n - 2:n]
                    d = (A[0, 0] - A[1, 1]) / 2
                    c = (A[1, 0] + A[0, 1]) / 2
                    sigma = A[1, 1] + d - np.sign(d) * np.sqrt(d**2 + c**2)

                if sigma != 0:
                    H[np.diag_indices_from(H[:n, :n])] -= sigma

                if simmetric:
                    sim_qr(H, n)
                else:
                    top_qr(H, n)

                if sigma != 0:
                        H[np.diag_indices_from(H[:n, :n])] += sigma

                counter += 1

            if abs(H[n - 1, n - 2]) < eps:
                eigenvalues.append(H[n-1, n-1])

                n -= 1
            else:
                z1, z2 = two_block_solution(H[n - 2:n, n - 2:n])

                eigenvalues.append(z1[0] + 1j * z1[1])
                eigenvalues.append(z2[0] + 1j * z2[1])

                n -= 2

    if info_mode:
        return np.array(eigenvalues[::-1]), counter
    
    return np.array(eigenvalues[::-1])


def naive_shur(A, iters=10):
    H = hessenberg(A)
    for _ in range(iters):
        q, R = top_qr(H)
        H = R @ q

    return np.diag(H)
