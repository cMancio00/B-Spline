from __future__ import annotations
import numpy as np
from numpy.linalg import qr
from numpy.linalg import solve
from  B_Spline import B_Spline
from matplotlib import pyplot as plt


def main():
    A  = B_Spline(
        knots=np.linspace(0,1,10+1),
        order=3
    ).compute_base().get_collocation_matrix()

    A = np.transpose(A)
    Q,R = qr(A,"complete")

    print("-----A----")
    print(np.shape(A))
    print("-----Q----")
    print(np.shape(Q))
    print("-----R----")
    print(np.shape(R))
    

    np.random.seed(1304)
    x = np.linspace(-3, 3, 600)
    #y = np.random.normal(3 + np.power(x,2), 1, 600)
    y = np.random.normal(np.sin(x),1,600)

    data = np.matrix([x, y]).T

    print("-----Data----")
    print(np.shape(data))

    c = np.transpose(Q) @ data
    c = c[0:np.shape(R)[1],:]
    print("-----c----")
    print(np.shape(c))
    print(c)

    x = solve(
        R[0:np.shape(R)[1],:],
        c
    )
    print("-----x----")
    print(np.shape(x))
    print(x)

    curve = A @ x






    plt.plot(data[:,0],data[:,1] , "o", label="data")
    plt.plot(curve[:,0],curve[:,1],"r-",label="fit")
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    main()
