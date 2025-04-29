import matplotlib.pyplot as plt
import mnumpy
import numpy as np
import time

def generateTest(n: int = 200) -> tuple[np.array, np.array]:
    A = np.random.rand(n, n)
    A = 0.5 * (A + A.T) + n * np.eye(n)
    b = np.random.rand(n)

    return (A, b)

def printResult(
        iter_cnt: int,
        max_iter: int,
        error: float,
        etalon_time: float,
        my_time: float
        ) -> None:
    
    print(f"Etalon time = {etalon_time}")
    print(f"My solution time = {my_time}")
    print(f"Error = {error}")
    print(f"Iterations taken = {iter_cnt} / {max_iter}")

def showErrorGraph(errors: list) -> None:
    plt.plot(errors)
    plt.xlabel("Iteration number")
    plt.ylabel("Error")
    plt.show()
    

def runTest(
        A:np.array,
        b: np.array,
        max_iter: int = 1000,
        tol: float = 1e-9
        ) -> None:
    
    start = time.time()
    my_sol, my_errors = mnumpy.symmetricZeidel(A, b, max_iter, tol)
    end = time.time()
    my_time = end - start

    start = time.time()
    sol_etalon = np.linalg.solve(A, b)
    end = time.time()
    etalon_time = end - start

    error = np.linalg.norm(sol_etalon - my_sol)
    printResult(len(my_errors), max_iter, error, etalon_time, my_time)
    showErrorGraph(my_errors)

def main() -> None:
    A, b = generateTest(450)
    runTest(A, b)

if __name__ == "__main__":
    main()