import scipy.integrate
import mnumpy
import numpy as np
import scipy

def compare_gauss(n: int, my_node: np.ndarray, my_weights: np.ndarray) -> None:
    ideal_node, ideal_weights = np.polynomial.legendre.leggauss(n)

    node_deviation = np.max(np.abs(np.sort(my_node) - ideal_node))
    weight_deviation = np.max(np.abs(np.sort(my_weights) - ideal_weights))

    print("=== NODE DEVIATION && WEIGHTS_DEVIATION ===")
    print("=== " + str(node_deviation) + " && " + str(weight_deviation) + " ===")

def test_case(func, a: int, b: int, n: int) -> None:
    print("================================================")
    print("=== RUN TEST: ===")

    ideal, _ = scipy.integrate.quad(func, a, b)
    mine_sol, my_node, my_weights = mnumpy.mgquadrate(func, a, b, n)
    print("=== IDEAL SOLUTION | MY SOLUTION | ERR ===")
    print("=== " + str(ideal) + " | " + str(mine_sol) + " | " + str(abs(ideal - mine_sol)) + " ===")

    compare_gauss(n, my_node, my_weights)
    print("================================================")

def main() -> None:
    a = -3
    b = 1
    n = 100
    func = lambda x: 3 * x ** 7 + 2 * x ** 4 + 12
    test_case(func, a, b, n)

    a = -2
    b = 2
    n = 80
    func = lambda x: np.exp(3 * x)
    test_case(func, a, b, n)

if __name__ == "__main__":
    main()
