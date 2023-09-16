import numpy as np
import matplotlib.pyplot as plt

def runge_function(x):
    return 1 / (1 + 25 * x**2)

def main():
# Genera dati casuali con rumore
    np.random.seed(0)
    x = np.linspace(0, 10, 100)
    
    x = np.linspace(-1, 1, 400)
    y_true = runge_function(x)
    y = y_true + np.random.normal(0, 0.1, len(x))

    plt.scatter(x, y_true, label='Dati originali')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()