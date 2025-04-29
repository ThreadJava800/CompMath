import numpy as np
from numba import jit

@jit(nopython=3)
def symmetricZeidel(
        A: np.array,
        b: np.array,
        max_iter: int,
        tol: float
        ) -> tuple[np.array, list]:
    
    assert(A.ndim == 2)
    assert(b.ndim == 1)

    res = np.zeros_like(b)
    err_arr = []

    for _ in range(max_iter):
        x_cpy = np.copy(res)

        for i in range(len(b)):
            part1 = np.dot(A[i, :i], res[:i])
            part2 = np.dot(A[i, i + 1:], res[i + 1:])
            x_cpy[i] = (b[i] - part1 - part2) / A[i, i]

        for i in range(len(b) - 1, -1, -1):
            part1 = np.dot(A[i, :i], x_cpy[:i])
            part2 = np.dot(A[i, i + 1:], x_cpy[i + 1:])
            x_cpy[i] = (b[i] - part1 - part2) / A[i, i]

        err = np.linalg.norm(A @ x_cpy - b)
        err_arr.append(err)
        
        if err < tol:
            return (x_cpy, err_arr)
        res = x_cpy

    return (res, err_arr)