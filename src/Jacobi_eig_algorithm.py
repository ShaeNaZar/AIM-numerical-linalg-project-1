import math
import numpy as np
from scipy.linalg import issymmetric


def do_rotation(A: np.ndarray, j: int, k: int, eps: float, old_J: np.ndarray, eigvals: bool):
    '''
    Implement givenc (jacobi) rotation to zero the A[j, k] and A[k, j] elements

    input: 
    A - matrix
    j, k - indices for element
    eps - eps for non-exact zero
    old_J - previous rotation matrix for cumulating rotations
    eigvals - if False old_J won't be updated

    output:
    A - new matrix with zero A[j, k] and A[k, j]
    J - final rotation matrix
    '''

    a_jk = A[j, k]
    if abs(a_jk) > eps:
        a_jj = A[j, j]
        a_kk = A[k, k]

        if abs(a_jj - a_kk) < eps:
            c = 1
            s = 0
        else:
            tau = (a_jj - a_kk) / (2 * a_jk)
            t = np.sign(tau) / (abs(tau) + math.sqrt(1 + tau**2))
            c = 1 / math.sqrt(1 + t**2)
            s = c * t

        J = np.array([[c, -s],[s, c]])

        A[[j, k], :] = J.T @ A[[j, k], :]
        A[:, [j, k]] = A[:, [j, k]] @ J

        if eigvals:
            old_J[:, [j, k]] = old_J[:, [j, k]] @ J

    new_J = old_J

    return A, new_J

def do_rotation_cycle(A: np.ndarray, old_J: np.ndarray, eps: float, eigvals: bool):
    '''
    Implement one cycle for each subdiagonal elements of A

    input:
    A - matrix
    old_J - previous rotation matrix for cumulating rotations
    eps - eps for non-exact zero
    eigvals - if False old_J won't be updated

    output:
    A - new matrix with zero non-diagonal elements after one cycle
    new_j - final rotation matrix
    '''

    n = A.shape[0]

    for j in range(1, n):
        for k in range(0, j):
            A, old_J = do_rotation(A, j, k, eps, old_J, eigvals)
    
    new_J = old_J

    return A, new_J

def Jacobi_algorithm(A: np.ndarray, J: np.ndarray=None, eps: float=1e-15, max_cylces: int=10, eigvals: bool=True):
    '''
    Implement Jacobi eigenvalue algorithm
    '''

    if not issymmetric(A, rtol=1e-5):
        raise ValueError('matrix must be symmetric!')
    
    n = A.shape[0]

    if eigvals:
        if J is None:
            J = np.eye(n)
        else:
            J = J.copy()
    else:
        J = None

    A = A.copy()
    A_for_norm = A.copy()
    np.fill_diagonal(A_for_norm, 0)
    norm_off = np.linalg.norm(A_for_norm, ord='fro')

    tol = eps * n**2
    cycle = 0
    while norm_off > tol and cycle < max_cylces:
        A, J = do_rotation_cycle(A, J, eps, eigvals)

        A_for_norm = A.copy()
        np.fill_diagonal(A_for_norm, 0)
        norm_off = np.linalg.norm(A_for_norm, ord='fro')
        cycle += 1

    return A, J, norm_off, cycle


if __name__ == '__main__':
    A = np.random.randn(4, 4)
    A = A.T + A
    res = Jacobi_algorithm(A)

    print('было:\n', A)
    print()
    print('стало:\n', res[0])
    print()
    print('с.в.:\n', res[1])
    print()
    print('off=', res[2])
