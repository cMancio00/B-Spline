from __future__ import annotations
import numpy as np
from numpy.linalg import qr
from numpy.linalg import solve
from  B_Spline import B_Spline
from HB_Spline import HB_Spline
from Collocable_Interface import Collocable
from matplotlib import pyplot as plt
from numpy.random import choice
from copy import deepcopy
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
        self.collocation_matrix = self.base.get_collocation_matrix()
        self.control_points = self.least_square_qr(self.collocation_matrix,self.data)
        self.curve = self.collocation_matrix @ self.control_points
        self.MSE()
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
    
    def find_bigger_contributors_sse(self)->tuple(int,int):
        indirect_sorted = np.argsort(self.sse)
        if indirect_sorted[-1] < indirect_sorted[-2]:         
            return indirect_sorted[-1], indirect_sorted[-2]
        return indirect_sorted[-2], indirect_sorted[-1]

    #TODO: La rifinitura potrebbe creare matrici singolari
    #Si dovrebbe mappare la rifinitura dal dominio dei dati al dominio del vettore dei nodi?

    def refine(self,range=None)->Model:
        if range is None:
            if(isinstance(self.base, B_Spline)):
                knot_id = np.argsort(self.sse)[-1]
                self.base.insert_knot(
                    self.curve[knot_id,0]
                )
            else:
                start_idx,stop_idx = self.find_bigger_contributors_sse()         
                self.base.refine(
                    (
                        self.curve[start_idx,0],
                        self.curve[stop_idx,0]
                    )
                )
        else:
            self.base.refine(range)

        self.collocation_matrix = self.base.get_collocation_matrix()
        self.fit()
        return self
    
    def iterative_refine(self)->Model:
        try:
            while True:
                old_model = deepcopy(self)
                self.refine()
                if(old_model.mse < self.mse):
                    self = deepcopy(old_model)
                    return self
        except np.linalg.LinAlgError:
            self = deepcopy(old_model)
            return self

    def plot(self, axes = None)->None:
        if axes is None:
            fig, axes = plt.subplots() 
        axes.plot(self.data[:, 0], self.data[:, 1], "bo", label="data")
        axes.plot(self.curve[:, 0], self.curve[:, 1], "r-", label="fit")
        #plt.plot(self.control_points[:,0],self.control_points[:,1],"x:g",label="control points")
        axes.legend(loc="best")
        print("MSE:"+"{:e}".format(self.mse))

    def MSE(self)->np.float32:
        dimension = np.shape(self.data)[0]
        self.sse = np.empty(dimension,dtype="float")
        for i in range(dimension):
            self.sse[i] = (self.data[i][0,1] - self.curve[i][0,1])**2
        self.mse = self.sse.sum() / dimension
        return self.mse


def main():
    def runge_function(x):
        return 1 / (1 + 25 * x**2)
    
    base_hb  = B_Spline(
        knots=np.linspace(-1,1,10+1),
        order=3
    )

    np.random.seed(1304)
    samples = np.shape(base_hb.compute_base().get_collocation_matrix())[1]
    #Addesso le B-Spline sono state calcolate e possiamo dichiarare le HB-Spline
    hb_a = HB_Spline(base_hb)
    hb_b = HB_Spline(base_hb)
    #Generazione dati per la funzione di runge (Dimensione per le HB-Spline)
    x = np.linspace(-1, 1, samples)
    y_true = runge_function(x)
    y = y_true + np.random.normal(0, 0.1, len(x))
    data_hb = np.matrix([x, y]).T
    
    b = Model(
    base=hb_b,
    data=data_hb
    )
    b.fit().iterative_refine()
    
    a = Model(
    base=hb_a,
    data=data_hb
    ).fit().refine((-0.25,0.25)).refine((-0.1,0.1))


    #Base B-Spline
    base  = B_Spline(
        knots= [-1,-1,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1,1],
        order=3
    )
    samples = np.shape(base.compute_base().get_collocation_matrix())[1]
    x = np.linspace(-1, 1, samples)
    y_true = runge_function(x)
    y = y_true + np.random.normal(0, 0.1, len(x))
    data = np.matrix([x, y]).T

    c = Model(
        base=base,
        data=data
    ).fit().iterative_refine()


    fig, ax = plt.subplots(3, 1, figsize=(16,12))

    a.plot(ax[0])
    ax[0].plot(x, y_true, "y-", label="real")
    ax[0].set_title("Raffinatura manuale (-0.25, 0.25), (-0.1, 0.1)"+" MSE:"+"{:e}".format(a.mse))

    b.plot(ax[1])
    ax[1].plot(x, y_true, "y-", label="real")
    ax[1].set_title("Raffinatura automatica"+" MSE:"+"{:e}".format(b.mse))

    c.plot(ax[2])
    ax[2].plot(x, y_true, "y-", label="real")
    ax[2].set_title("Raffinatura automatica B-Spline"+" MSE:"+"{:e}".format(c.mse))

    plt.suptitle("Confronto tra metodi")
    plt.show()


if __name__ == "__main__":
    main()
