import mnumpy
import numpy as np

def compareSolutions(x1: np.array, x2: np.array) -> float:
    assert(x1.ndim == x2.ndim)

    if x1.ndim == 1:
        return mnumpy.__l2Norm(x1 - x2)
    if x2.ndim == 2:
        return mnumpy.__frobeniusNorm(x1 - x2)
    
    raise Exception("Unsupported dimension. ndim = {x1.ndim}")

def numpySolver1(A: np.array, b: np.array) -> np.array:
    if A.shape[0] == A.shape[1]:
        return np.linalg.solve(A, b)
    
    Q, R = np.linalg.qr(A)
    return np.linalg.solve(R, Q.T @ b)

def numpySolver2(A: np.array, b: np.array) -> np.array:
    if A.shape[0] == A.shape[1]:
        return np.linalg.lstsq(A, b)
    
    Q, R = np.linalg.qr(A)
    return np.linalg.lstsq(R, Q.T @ b)

def test1() -> None:
    print("================TEST 1================")

    A = np.array([[3., 2., -5.], [2., -1., 3.], [1., 2., 3.]])
    b = np.array([-1., 13., 9.])
    
    _, _, qr_sol = mnumpy.solveLinearSystem(A, b)
    np1Sol = numpySolver1(A, b)
    np2Sol = numpySolver2(A, b)[0]

    print("||mine - np.solve|| =", compareSolutions(qr_sol, np1Sol))
    print("||mine - np.lstsq|| =", compareSolutions(qr_sol, np2Sol))

def test2() -> None:
    print("================TEST 2================")

    A = np.array([[3., 2.], [2., -1.], [3., 2.]])
    b = np.array([-1., 13., -1.])

    _, _, qr_sol = mnumpy.solveLinearSystem(A, b)
    np1Sol = numpySolver1(A, b)
    np2Sol = numpySolver2(A, b)[0]

    print("||mine - np.solve|| =", compareSolutions(qr_sol, np1Sol))
    print("||mine - np.lstsq|| =", compareSolutions(qr_sol, np2Sol))

def main() -> None:
    test1()
    test2()


if __name__ == "__main__":
    main()
