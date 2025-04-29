import numpy as np

def __getWeights(n: int) -> [np.ndarray, np.ndarray]:
    # Голуб-Вельш
    beta = np.array([i / np.sqrt(4 * i * i - 1) for i in range(1, n)])
    three_diag_sym_matr = np.diag(beta, 1) + np.diag(beta, -1)

    eigvals, eigvecs = np.linalg.eigh(three_diag_sym_matr)
    return eigvals, 2 * (eigvecs[0, :] ** 2)

def mgquadrate(func, a: float, b: float, n: int) -> [float, np.ndarray, np.ndarray]:
    eigvals, weights = __getWeights(n)
    x_ab = (b - a) * eigvals / 2 + (a + b) / 2 # переводим из [-1;1] к [a;b]
    w_ab = (b - a) * weights / 2

    return np.sum(w_ab * func(x_ab)), w_ab, x_ab
