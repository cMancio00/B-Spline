from __future__ import annotations
import numpy as np
from numpy.linalg import qr
from numpy.linalg import solve
from  B_Spline import B_Spline
from HB_Spline import HB_Spline
from Collocable_Interface import Collocable
from matplotlib import pyplot as plt
from numpy.random import choice
from time import time


class Model:
    def __init__(self,base:Collocable,data:np.ndarray) -> None:
        self.base = base
        self.collocation_matrix = self.base.get_collocation_matrix()
        # self.data = self.put_data_in_proper_dimension(data)
        self.data = data
        self.curve = []
        self.control_points = []

    #TODO: rivedere la seguente funzione
    def put_data_in_proper_dimension(self,data:np.ndarray)->np.ndarray:
        correct_dimension = np.shape(self.collocation_matrix)[1]
        correct_data = np.empty([correct_dimension,2])
        for i in range(correct_dimension):
            idx = choice(correct_dimension)
            correct_data[i] = data[idx]
        #correct_data.sort(kind="quicksort")
        print(correct_data)
        print(np.shape(correct_data))
        return correct_data
        

    def fit(self)->Model:
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
        plt.plot(self.data[:,0],self.data[:,1] , "bo", label="data")
        plt.plot(self.curve[:,0],self.curve[:,1],"r-",label="fit")
        plt.plot(self.control_points[:,0],self.control_points[:,1],"x:g",label="control points")
        plt.legend(loc="best")

    



def main():
    base  = B_Spline(
        knots=np.linspace(-3,3,10+1),
        order=3
    )

    hb = HB_Spline(base)
    
    np.random.seed(1304)
    samples = np.shape(base.compute_base().get_collocation_matrix())[1]
    x = np.linspace(-3, 3, samples)
    y1 = np.random.normal(5 + np.power(x,2), 1, samples)
    y2= np.random.normal(np.sin(x),1,samples)

    data1 = np.matrix([x, y1]).T
    data2 = np.matrix([x, y2]).T

    Model(
        base=hb,
        data=data1
    ).fit().plot()

    # Model(
    #     base=base,
    #     data=data2
    # ).fit().plot()

    plt.show()



if __name__ == "__main__":
    main()
