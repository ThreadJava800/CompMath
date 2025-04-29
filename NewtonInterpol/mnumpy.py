import numpy as np

def calcDividedDiffs(
        points: np.array,
        func_values: np.array
    ) -> np.array:
    assert (len(points) == len(func_values))

    poly_coeffs = np.array(func_values)
    for i in range(1, len(points)):
        poly_coeffs[i:] = \
            (poly_coeffs[i:] - poly_coeffs[i - 1]) / (points[i:] - points[i - 1])
            
    return poly_coeffs

def calcNewtonPolynomialInPoint(
        points: np.array,
        poly_coeffs: np.array,
        calc_point: float
    ) -> float:
    assert(len(points) == poly_coeffs.shape[0])

    res = np.zeros_like(calc_point)
    for k in range(len(points)):
        term = poly_coeffs[k]

        for j in range(k):
            term *= (calc_point - points[j])
        res += term
        
    return res

def calcNewtonDerivativeInPoint(
        points: np.array,
        poly_coeffs: np.array,
        calc_point: float
    ) -> float:
    assert(len(points) == poly_coeffs.shape[0])

    res = np.zeros_like(calc_point)
    for k in range(1, len(points)):
        term = poly_coeffs[k]
        prod = np.ones_like(calc_point)

        for j in range(k):
            prod *= (calc_point - points[j])

        res += term * np.sum(
            prod, axis=0
        )
        
    return res

def calcNewtonDerivativePoints(
        points: np.array,
        poly_coeffs: np.array,
        calc_point: np.array
    ) -> np.array:
    res = []
    for cp in calc_point:
        res.append(calcNewtonDerivativeInPoint(points, poly_coeffs, cp))
    return res

def calcNewtonPolynomialPoints(
        points: np.array,
        poly_coeffs: np.array,
        calc_point: np.array
    ) -> np.array:
    res = []
    for cp in calc_point:
        res.append(calcNewtonPolynomialInPoint(points, poly_coeffs, cp))
    return res
