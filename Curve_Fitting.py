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
    def __init__(self,base:HB_Spline,data:np.ndarray) -> None:
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
        self.collocation_matrix = self.base.get_collocation_matrix()
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
    
    #TODO: La rifinitura potrebbe creare matrici singolari
    #Si dovrebbe mappare la rifinitura dal dominio dei dati al dominio del vettore dei nodi?
    
    def refine(self,range:tuple)->Model:
        self.base.refine(range)
        self.collocation_matrix = self.base.get_collocation_matrix()
        self.fit()
        return self

    def plot(self):
        plt.plot(self.data[:,0],self.data[:,1] , "bo", label="data")
        plt.plot(self.curve[:,0],self.curve[:,1],"r-",label="fit")
        #plt.plot(self.control_points[:,0],self.control_points[:,1],"x:g",label="control points")
        plt.legend(loc="best")

    




def main():
    
    base  = B_Spline(
        knots=np.linspace(-1,1,10+1),
        order=3
    )

    hb = HB_Spline(base)
    
    np.random.seed(1304)

    samples = np.shape(base.compute_base().get_collocation_matrix())[1]

    x = np.linspace(0, 10, samples)
    y= x + np.random.normal(0, 1, samples)

    plt.plot(x,y , "bo", label="data")
    plt.legend(loc="best")
    plt.show()

    x = np.linspace(-3, 3, samples)
    y = np.random.normal(3 + np.power(x,2), 1, samples)

    plt.plot(x,y , "bo", label="data")
    plt.legend(loc="best")
    plt.show()

    x = np.linspace(0, 10, samples)
    y= np.random.normal(np.sin(x),1,samples)

    plt.plot(x,y , "bo", label="data")
    plt.legend(loc="best")
    plt.show()

    y = np.sin(x) + np.sin(2 * x) + np.sin(3 * x) + np.random.normal(0, 1, samples)

    plt.plot(x,y , "bo", label="data")
    plt.legend(loc="best")
    plt.show()

    def runge_function(x):
        return 1 / (1 + 25 * x**2)
    x = np.linspace(-1, 1, samples)
    y_true = runge_function(x)
    y = y_true + np.random.normal(0, 0.1, len(x))
    plt.plot(x,y , "bo", label="data")
    plt.legend(loc="best")
    plt.show()

    y = y_true + np.random.normal(0, 1, len(x))
    plt.plot(x,y , "bo", label="data")
    plt.legend(loc="best")
    plt.show()

    data = np.matrix([x, y]).T

    a = Model(
        base=hb,
        data=data
    ).fit()

    a.plot()

    plt.show()

    a.refine((-0.25,0.25))
    a.plot()
    plt.show()

    a.refine((-0.1,0.1))
    a.plot()
    plt.show()



if __name__ == "__main__":
    main()
