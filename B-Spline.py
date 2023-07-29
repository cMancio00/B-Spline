import numpy as np
import matplotlib.pyplot as plt


class B_Spline:
    def __init__(self,extended_knot_vector,order):
        self.extended_knot_vector = extended_knot_vector
        self.order = order
        self.n = np.shape(self.extended_knot_vector)[0] - self.order
        self.L = self.n-self.order+2
        
    def __str__(self) -> str:
        return \
        f"Order: {self.order}\n" + \
        f"Number of B-Spline: {self.n + 1}\n" + \
        f"Extended Knot Vector lenght: {self.n + self.order}\n" + \
        f"Left auxiliary knots: {(self.n + self.order - self.L)//2}\n" + \
        f"Right auxiliary knots: {(self.n + self.order- self.L)//2}\n" + \
        f"n: {self.n}\n" + \
        f"Number of intervals: {self.L}\n"

    
    def compute_base(self, t:np.array):
        self.base = np.empty((self.n, np.shape(t)[0]))
        self.t = t
        for i in range(self.n+self.order-1):
            for j in range(np.shape(t)[0]):
                if  self.extended_knot_vector[i] <= t[j] < self.extended_knot_vector[i+1]:
                    self.base[i, j] = 1
                else:
                    self.base[i, j] = 0
    
    def plot_base(self):
        fig, axs = plt.subplots(nrows=self.n, ncols=1)
        fig.suptitle('B-Spline Base')
        for i in range(self.n):
            axs[i].plot(self.t, self.base[i])
            axs[i].grid()
        plt.show()


def main():
    T = np.linspace(0, 4, 5)
    print(T)
    b = B_Spline(T, 1)
    print(b)
    b.compute_base(np.linspace(0, 4, 1000))
    print(b.base)
    b.plot_base()

if __name__ == "__main__":
    main()