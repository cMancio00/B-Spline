from B_Spline import B_Spline
import numpy as np
from Control_Points import Control_Points
from matplotlib import pyplot as plt
from time import time

class Curve():

    def __init__(self,collocation_matrix:B_Spline) -> None:
        self.order = collocation_matrix.order
        try:
            collocation_matrix.get_collocation_matrix()
            print("Basis already computed")
        except AttributeError:
            print("Compunting base...")
            start = time()
            collocation_matrix.compute_base()
            stop = time()
            print(f"Done\nComputed in {stop - start}")
        finally:
            self.collocation_matrix = collocation_matrix.get_collocation_matrix()
            number_of_contol_points = np.shape(self.collocation_matrix)[0]
            print(f"Curve needs {number_of_contol_points} control points")

        self.control_points =  Control_Points(number_of_contol_points).build_control_polygon().control_points

    
    def compute_curve(self) :
        self.curve_points = self.control_points @ self.collocation_matrix
        return self
    
       


    def plot_curve(self) -> None:
        print(np.shape(self.curve_points))
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid()
        plt.plot(self.control_points[0],self.control_points[1],':rx')
        plt.plot(self.curve_points[0],self.curve_points[1])
        plt.grid(True)
        plt.title(f"B-Splie Curve of order {self.order} and {np.shape(self.control_points)[1]} control points")
        plt.xlabel(f"{np.shape(self.curve_points)[1]} points plotted")

        plt.show()

def main():

    T = np.linspace(0, 1, 10 + 1)
    #T = [0,0,0,0.25,0.5,1,1,1]
    basis = B_Spline(T, 3)
    curve = Curve(basis)
    curve.compute_curve().plot_curve()



if __name__ == "__main__":
    main()