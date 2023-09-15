from __future__ import annotations
import numpy as np
from numpy.linalg import qr
from numpy.linalg import solve
from  B_Spline import B_Spline
from HB_Spline import HB_Spline
from Collocable_Interface import Collocable
from matplotlib import pyplot as plt
from time import time


class Model:
    def __init__(self,base:Collocable,data:np.ndarray) -> None:
        self.base = base
        self.data = data
        self.collocation_matrix = []
        self.curve = []
        self.control_points = []

    def fit(self)->Model:
        self.collocation_matrix = self.base.compute_base().get_collocation_matrix()
        self.control_points = self.least_square_qr(self.collocation_matrix,self.data)
        self.curve = self.collocation_matrix @ self.control_points
        return self

    def least_square_qr(self,A:np.ndarray,b:np.ndarray)->np.ndarray:
        A = np.transpose(A)
        self.collocation_matrix = A
        Q,R = qr(A,"complete")
        c = np.transpose(Q) @ b
        c = c[0:np.shape(R)[1],:]
        x = solve(
        R[0:np.shape(R)[1],:],
        c
        )
        return x

    def plot(self):
        plt.plot(self.data[:,0],self.data[:,1] , "o", label="data")
        plt.plot(self.curve[:,0],self.curve[:,1],"r-",label="fit")
        plt.legend(loc="best")

    



def main():
    base  = B_Spline(
        knots=np.linspace(-3,3,10+1),
        order=3
    )
    np.random.seed(1304)
    x = np.linspace(-3, 3, 600)
    y = np.random.normal(3 + np.power(x,2), 1, 600)
    #y = np.random.normal(np.sin(x),1,600)

    data = np.matrix([x, y]).T
    
    A = HB_Spline(
        B_Spline(np.linspace(-3,3,10+1),3)
    ).refine((-2,2)).refine((-1,1)).get_hierarchical_basis().hb_basis

    A = A[:,200:800]

    Model(
        base=base,
        data=data
    ).fit().plot()

    plt.show()



if __name__ == "__main__":
    main()
