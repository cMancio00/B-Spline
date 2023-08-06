import numpy as np

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
    
        
    def __str__(self) -> str:
        return \
        f"Order: {self.order}\n" + \
        f"Number of Basis functions: {self.number_basis_function}\n" + \
        f"Extended Knot Vector lenght: {len(self.knots)}\n"
   
    def get_base(self) -> np.ndarray:
        if not self.basis:
            raise AttributeError("B-Spline base has not yet been calculated") 
        return self.basis[-1]
    
    def omega(self,breakpoint:float, i:int,r:int) -> float:
        # TODO: Il controllo if breakpoint < self.knots[i+r-1] (previsto in letteratura),
        # può comunque generare zero in alcuni casi
        # del calcolo (1 - omega(breakpoint,i+1,r))
        if (self.knots[i+r-1] - self.knots[i]) != 0:
            return (breakpoint - self.knots[i]) / (self.knots[i+r-1] - self.knots[i])
        else:
            return 0
  
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
                        self.omega(breakpoint,i,r) \
                        * base[i][j] \
                        + (1 - self.omega(breakpoint,i+1,r)) \
                        * base[i+1][j]
                #TODO: Controllare se è necessario effettuare una deep copy
                base = base[:-1,:]
                self.basis.append(base)

    






            




