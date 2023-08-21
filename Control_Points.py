from matplotlib import pyplot as plt
import numpy as np

class Control_Points:
    def __init__(self,number:int) -> None:
        self.number = number
        self.control_points = None
        
    def build_control_polygon(self):
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid()
        gpoints = plt.ginput(self.number)
        plt.close()
        self.control_points = np.transpose(np.asarray(gpoints))
        return self

        
        