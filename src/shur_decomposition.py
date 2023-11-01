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

    mu = 1 / np.sqrt(2 * beta**2 - 2 * v[0] * beta)

    w = v.copy()
    w[0] = w[0] - beta

    res = np.identity(len(w)) - 2 *(mu**2) *np.outer(w, w) 

    return res


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
    r = np.sqrt(x[0]**2 + x[1]**2)
    cos = x[0] / r
    sin = x[1] / r

    return np.array([[cos, sin], [-sin, cos]])


def top_qr(A):
    n = len(A)

    Q = np.identity(n)
    R = A.copy()

    for i in range(n - 1):
        if R[i + 1, i] == 0:
            continue

        G = givenc(R[i:i+2, i])
        
        R[i:i+2, :] = G @ R[i:i+2, :]
        Q[i:i+2, :] = G @ Q[i:i+2, :]

    return Q.T, R


def eigenvalues(A, eps=1e-12, method="naive"):
    H = hessenberg(A)
    n = len(A)
    counter = 0
    eigenvalues = []

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
                    if eigenvalues:
                        sigma = abs(eigenvalues[-1])
                        sigma = abs((abs(H[n - 1, n - 1]) - abs(sigma))) * np.sign(H[n - 1, n - 1])
                        H[np.diag_indices_from(H[:n, :n])] -= sigma

                
                q, R = top_qr(H[:n, :n])
                H[:n, :n] = R @ q

                if method == "simple":
                    if eigenvalues:
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

    print(f"num iterations: {counter}")
    return np.array(eigenvalues[::-1])

def naive_shur(A, iters=10):
    H = hessenberg(A)
    for _ in range(iters):
        q, R = top_qr(H)
        H = R @ q

    return np.diag(H)
