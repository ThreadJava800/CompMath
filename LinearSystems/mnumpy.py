import math
import numpy as np

def __scalarProduct(a: np.array, b: np.array) -> float:
    assert(a.ndim == 1)
    assert(b.ndim == 1)
    assert(a.size == b.size)

    product = 0.0
    for i in range(a.size):
        product += a[i] * b[i]
    
    return product

def __l2Norm(x: np.array) -> float:
    assert(x.ndim == 1)

    norm = 0.0
    for i in range(x.size):
        norm += x[i] * x[i]

    return math.sqrt(norm)

def __gramSchmidtProcess(A: np.array) -> np.array:
    assert(A.ndim == 2)

    Q = np.array(A)
    col_cnt = A.shape[1]

    for i in range(col_cnt):
        q = Q[:, i] / __l2Norm(Q[:, i])

        for k in range(i + 1, col_cnt):
            Q[:, k] = Q[:, k] - q * __scalarProduct(q, Q[:, k])

    return Q

def __normalizeMatrix(Q: np.array) -> np.array:
    assert(Q.ndim == 2)

    col_cnt = Q.shape[1] 

    for i in range(col_cnt):
        Q[:, i] = Q[:, i] / __l2Norm(Q[:, i])

    return Q

def __getUpperTriangleHalf(Q: np.array, A: np.array) -> np.array:
    assert(Q.ndim == 2)
    assert(A.ndim == 2)

    return Q.T @ A
    # return np.linalg.inv(Q) @ A

def __backSubstitution(A: np.array, b: np.array) ->np.array:
    # A = U, b = y
    sol = np.zeros_like(b)
    sol[-1] = b[-1] / A[-1, -1]
    for i in range(b.shape[0] - 2, -1, -1):
        sum = b[i]

        for j in range(b.shape[0] - 1, i, -1):
            sum = sum - A[i, j] * sol[j]

        sol[i] = sum / A[i, i]
    
    return sol
    

def solveLinearSystem(A: np.array, b: np.array) -> tuple[np.array, np.array, np.array]:    
    Q = __gramSchmidtProcess(A)
    Q = __normalizeMatrix(Q)
    R = __getUpperTriangleHalf(Q, A)
    x = __backSubstitution(R, Q.T @ b)

    print(x)


