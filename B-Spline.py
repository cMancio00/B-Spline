import numpy as np
import matplotlib.pyplot as plt


class B_Spline:
    def __init__(self,extended_knot_vector,order):
        self.extended_knot_vector = extended_knot_vector
        self.order = order
        self.n = np.shape(self.extended_knot_vector)[1] - self.order
        self.L = self.n-self.order+2
        self.base = np.zeros((self.n+1, 1))
    
    def __str__(self) -> str:
        return \
        f"Order: {self.order}\n" + \
        f"Number of B-Spline: {self.n +1}\n" + \
        f"Extended Knot Vector lenght: {self.n + self.order}\n" + \
        f"Left auxiliary knots: {(self.n + self.order - self.L)//2}\n" + \
        f"Right auxiliary knots: {(self.n + self.order- self.L)//2}\n" + \
        f"n: {self.n}\n" + \
        f"Number of intervals: {self.L}\n"


def main():
    T = np.zeros((1, 6))
    b = B_Spline(T, 3)
    print(b)
    print(b.base)

if __name__ == "__main__":
    main()