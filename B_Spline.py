from __future__ import annotations
import numpy as np
from matplotlib import pyplot as plt

class B_Spline:
    # TODO: Implementare il builder
    def __init__(self,knots,order) -> None:
        # TODO: Eventualmente imporre la condizione che il vettore knots sia crescente
        self.knots = knots
        self.order = order
        if len(self.knots) - 1 < self.order:
            raise ValueError("Order is too high for the number of knots")
        self.number_basis_function = len(self.knots) - self.order
        self.t = np.linspace(self.knots[0],self.knots[-1],1000)
        self.basis = list()
        self.domain_range = list()
    
        
    def __str__(self) -> str:
        return \
        f"Order: {self.order}\n" + \
        f"Number of Basis functions: {self.number_basis_function}\n" + \
        f"Extended Knot Vector lenght: {len(self.knots)}\n"
    
    def get_valid_abscissa(self):
        for index,value in enumerate(self.t):
            if(value >= self.knots[self.order - 1]):
                start = index
                break
        for index,value in enumerate(self.t):
            if(value >= self.knots[self.number_basis_function]):
                stop = index - 1
                break
        return start, stop
   
    def get_base(self) -> np.ndarray:
        if not self.basis:
            raise AttributeError("B-Spline base has not yet been calculated") 
        return self.basis[-1]
    
    def get_collocation_matrix(self) -> np.ndarray:
        collocation_matrix = self.get_base()
        start,stop = self.get_valid_abscissa()
        return collocation_matrix[:,start : (stop +1)]
    
    def omega(self,breakpoint:float, i:int,r:int) -> float:
        # TODO: Il controllo if breakpoint < self.knots[i+r-1] (previsto in letteratura),
        # può comunque generare zero in alcuni casi
        # del calcolo (1 - omega(breakpoint,i+1,r))
        if (self.knots[i+r-1] - self.knots[i]) != 0:
            return (breakpoint - self.knots[i]) / (self.knots[i+r-1] - self.knots[i])
        else:
            return 0
  
    def compute_base(self) -> B_Spline:
        max_basis_function = len(self.knots) - 1
        base = np.empty((max_basis_function,len(self.t)))

        for i in range(np.shape(base)[0]):
            for j,breakpoint in enumerate(self.t):
                if self.knots[i] <= breakpoint < self.knots[i+1]:
                    base[i][j] = 1
                else:
                    base[i][j] = 0
        self.basis.append(base)

        if self.order != 1:

            for r in range(2,self.order+1):

                for i in range(np.shape(base)[0] - 1):
                    for j,breakpoint in enumerate(self.t):
                        base[i][j] = \
                        self.omega(breakpoint,i,r) \
                        * base[i][j] \
                        + (1 - self.omega(breakpoint,i+1,r)) \
                        * base[i+1][j]
                #TODO: Controllare se è necessario effettuare una deep copy
                base = base[:-1,:]
                self.basis.append(base)
            
        return self

    def compute_basis_range(self) -> B_Spline:
        for i in range(self.number_basis_function):
            base_range = {
                "start_idx" : i,
                "stop_idx" : i + self.order
            }
            self.domain_range.append(base_range)
        return self

def main():
    A = [0,0,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1,1]
    T = np.linspace(0, 1, 10 + 1)
    base  = B_Spline(T, 3)
    base.compute_base()
    print(base.number_basis_function)
    print(base.domain_range)


    for i in range(np.shape(base.get_base() )[0] ):
        plt.plot(base.t, base.get_base()[i][:])
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

    






            




