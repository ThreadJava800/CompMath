import mnumpy
import numpy as np

def main():
    # A = np.array([[1., 2., 3.], [3., 2., 1.]])
    A = np.array([[3., 2., -5.], [2., -1., 3.], [1., 2., -1.]])
    b = np.array([-1., 13., 9.])
    mnumpy.solveLinearSystem(A, b)


if __name__ == "__main__":
    main()
