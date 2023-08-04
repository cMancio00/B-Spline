import numpy as np

class B_Spline:
    def __init__(self,knots,order):
        # TODO: Guardare il conteggio degli intervalli e se il vettore dei nodi è valido
        self.knots = knots
        self.order = order
        self.number_base_function = len(self.knots) - self.order
        self.L = self.number_base_function-self.order+1
        self.n = self.number_base_function - 1
        self.t = np.linspace(self.knots[0],self.knots[-1],1000)
        self.basis = list()
        
    def __str__(self) -> str:
        return \
        f"Order: {self.order}\n" + \
        f"Number of Basis functions: {self.number_base_function}\n" + \
        f"Extended Knot Vector lenght: {len(self.knots)}\n" + \
        f"Left auxiliary knots: {self.order - 1}\n" + \
        f"Right auxiliary knots: {self.order - 1}\n" + \
        f"Number of intervals: {self.L}\n"
   
    def get_base(self) -> np.ndarray:
        if not self.basis:
            raise AttributeError("B-Spline base has not yet been calculated") 
        return self.basis[-1]
  
    def compute_base(self) -> None:
        base = np.empty((self.number_base_function,len(self.t)))

        for i in range(np.shape(base)[0]):
            for j,breakpoint in enumerate(self.t):
                if self.knots[i] <= breakpoint < self.knots[i+1]:
                    base[i][j] = 1
                else:
                    base[i][j] = 0
        self.basis.append(base)


        for i in range(np.shape(base)[0] - 1):
            for j,breakpoint in enumerate(self.t):
                base[i][j] = \
                (breakpoint - self.knots[i]) / (self.knots[i+2-1] - self.knots[i]) \
                * base[i][j] \
                + (self.knots[i+2] - breakpoint) / (self.knots[i+2] - self.knots[i+1]) \
                * base[i+1][j]

        base = base[:-1,:]
        self.basis.append(base)

    






            




