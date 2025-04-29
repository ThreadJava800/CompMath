import matplotlib.pyplot as plt
import mnumpy
import numpy as np

def drawNewtonGraph(func, func_deriv, dim: int) -> None:
    x_scale = np.linspace(-np.pi, np.pi, dim)
    y_scale = func(x_scale)

    x_cheba = np.cos((2 * np.arange(dim) + 1) / (2 * dim) * np.pi) * np.pi
    y_cheba = func(x_cheba)

    func_points = np.linspace(-np.pi, np.pi, 1000)

    coef_scale = mnumpy.calcDividedDiffs(x_scale, y_scale)
    poly_scale = mnumpy.calcNewtonPolynomialPoints(x_scale, coef_scale, func_points)
    deriv_poly_scale = mnumpy.calcNewtonDerivativePoints(x_scale, coef_scale, func_points)

    coef_cheba = mnumpy.calcDividedDiffs(x_cheba, y_cheba)
    poly_cheba = mnumpy.calcNewtonPolynomialPoints(x_cheba, coef_cheba, func_points)
    deriv_poly_cheba = mnumpy.calcNewtonDerivativePoints(x_cheba, coef_cheba, func_points)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # normal
    axes[0, 0].plot(func_points, func(func_points), 'k-', label='f(x)')
    axes[0, 0].plot(func_points, poly_scale, 'r--', label='Равномерная сетка')
    axes[0, 0].scatter(x_scale, y_scale, color='red', marker='o')
    axes[0, 0].set_title('Равномерная интерполяция')
    axes[0, 0].legend()

    axes[0, 1].plot(func_points, func(func_points), 'k-', label='f(x)')
    axes[0, 1].plot(func_points, poly_cheba, 'b--', label='Чебышевская сетка')
    axes[0, 1].scatter(x_cheba, y_cheba, color='blue', marker='o')
    axes[0, 1].set_title('Чебышевская интерполяция')
    axes[0, 1].legend()

    # derivatives
    axes[1, 0].plot(func_points, func_deriv(func_points), 'k-', label="f'(x)")
    axes[1, 0].plot(func_points, deriv_poly_scale, 'r--', label="L'_n (Равномерная)")
    axes[1, 0].set_title('Производная - равномерная')
    axes[1, 0].legend()

    axes[1, 1].plot(func_points, func_deriv(func_points), 'k-', label="f'(x)")
    axes[1, 1].plot(func_points, deriv_poly_cheba, 'b--', label="L'_n (Чебышев)")
    axes[1, 1].set_title('Производная - чебышевская')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()




def test_1() -> None:
    func = lambda x: np.sin(2 * x) + np.cos(3 * x)
    func_deriv = lambda x: 2 * np.cos(2 * x) - 3 * np.sin(3 * x)
    drawNewtonGraph(func, func_deriv, 10)


def main() -> None:
    test_1()

if __name__ == "__main__":
    main()
