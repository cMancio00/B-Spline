import numpy as np

class B_Spline:
    def __init__(self,knots,order):
        # TODO: Guardare il conteggio degli intervalli e se il vettore dei nodi è valido
        self.knots = knots
        self.order = order
        self.number_basis_function = len(self.knots) - self.order
        self.t = np.linspace(self.knots[0],self.knots[-1],1000)
        self.basis = list()
        
    def __str__(self) -> str:
        return \
        f"Order: {self.order}\n" + \
        f"Number of Basis functions: {self.number_basis_function}\n" + \
        f"Extended Knot Vector lenght: {len(self.knots)}\n"
   
    def get_base(self) -> np.ndarray:
        if not self.basis:
            raise AttributeError("B-Spline base has not yet been calculated") 
        return self.basis[-1]
  
    def compute_base(self) -> None:
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
                        (breakpoint - self.knots[i]) / (self.knots[i+r-1] - self.knots[i]) \
                        * base[i][j] \
                        + (self.knots[i+r] - breakpoint) / (self.knots[i+r] - self.knots[i+1]) \
                        * base[i+1][j]

                base = base[:-1,:]
                self.basis.append(base)

    






            




