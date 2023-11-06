import numpy as np
from scipy.linalg import hessenberg


class Bisection:
    @staticmethod
    def _check(matrix: np.array, rtol=1e-05, atol=1e-08) -> None:
        if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0 or not np.allclose(matrix, matrix.T, rtol, atol):
            raise Exception("Ошибка матрицы")

    def __init__(self, matrix: np.array, eps = np.exp(1) ** -20):
        A = hessenberg(matrix)
        Bisection._check(matrix)
        self.matrix = A
        self.len = len(self.matrix)
        self.diag = self.matrix.diagonal()
        self.upper_diag = self.matrix.diagonal(1)
        self.eps = eps
        self.eigen_values: list[float] = []
        self._iter = 0

    def inertia(self, sigma: float=0) -> int:
        d = self.diag[0] - sigma
        n_inert = 0 + (d < 0)
        for i in range(self.len - 1):
            d = (self.diag[i + 1] - sigma) - self.upper_diag[i] ** 2 / d
            n_inert += (d < 0)
        return n_inert
    def _find_eigen(self, start: float, end: float, iter=1) -> None:
        self._iter = max(self._iter, iter)
        first = self.inertia(start)
        second = self.inertia(end)
        if second - first == 0:
            return
        if np.abs(end - start) < 2 * self.eps:
            self.eigen_values.append(start)
            return
        middle = (start + end) / 2
        middle_inertia = self.inertia(middle)
        if np.abs(middle_inertia - first) > 0:
            self._find_eigen(start, middle, iter + 1)
        if np.abs(second - middle_inertia) > 0:
            self._find_eigen(middle, end, iter + 1)

    def _reset(self):
        self.eigen_values = []
        
    def bisection(self) -> (np.array, int,):
        norm = np.linalg.norm(self.matrix, ord=np.inf)
        mini = -norm
        maxi = norm
        self._find_eigen(mini, maxi)
        result = np.array(self.eigen_values)
        iter = self._iter
        self._reset()
        return result, iter
    


def eigenvalues(A):
    bs = Bisection(A)
    return bs.bisection()[0]