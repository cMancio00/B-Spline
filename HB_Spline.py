from __future__ import annotations
import numpy as np
from B_Spline import B_Spline
from matplotlib import pyplot as plt


class HB_Spline():
    def __init__(self,mother:B_Spline) -> None:
        self.mother = mother.compute_base()
        self.domains = [{"start":mother.knots[0],"stop":mother.knots[-1],"knots":len(mother.knots)}]
        self.vectors = [{"knots":mother.knots,"level":0}]
        self.level_basis = [{"basis":mother.get_base(),"level":0}]

    def refine(self,range:tuple):
        self.add_domain(range)
        self.add_vector()
        self.compute_level_base()
    
    def add_domain(self,range:tuple) -> None:
        start_idx,stop_idx = self.find_closest_range(range)
        domain = \
        {
            "start": self.vectors[-1]["knots"][start_idx],
            "stop": self.vectors[-1]["knots"][stop_idx],
            "knots": ((self.domains[-1]["knots"]) * 2) - 1
        }
        self.domains.append(domain)

    def find_closest_range(self,range:tuple) -> tuple[int,int]:
        start_idx = (np.abs(self.vectors[-1]["knots"] - range[0])).argmin()
        stop_idx = (np.abs(self.vectors[-1]["knots"] - range[1])).argmin()
        return start_idx, stop_idx
    
    def add_vector(self) -> None:
        knots = np.linspace(
            self.domains[0]["start"],
            self.domains[0]["stop"],
            self.domains[-1]["knots"],
            endpoint=True
        )
        vector = {
            "knots": knots,
            "level":len(self.vectors)
        }
        self.vectors.append(vector)

    def compute_level_base(self) -> None:
        base = B_Spline(
            knots=self.vectors[-1]["knots"],
            order = self.mother.order
        ).compute_base().get_base()

        level_base = {
            "basis":base,
            "level":self.vectors[-1]["level"]
        }
        self.level_basis.append(level_base)

    def plot_level_basis(self) -> None:
        for level in range (len(self.level_basis)):
            for i in range (np.shape(self.level_basis[level]["basis"])[0]):
                plt.subplot(len(self.level_basis),1,level+1)
                plt.title(f"Level {level}")
                plt.grid(True)
                plt.plot(
                    np.linspace(
                    start= self.mother.t[0],
                    stop= self.mother.t[-1],
                    num= self.mother.t.size,
                    endpoint=True
                    ),
                    self.level_basis[level]["basis"][i,:])
        plt.suptitle("All basis")



def main():
    T = np.linspace(0, 1, 7 + 1)
    mother = B_Spline(T, 3)
    hb = HB_Spline(mother)
    hb.refine((0.2,0.8))
    hb.refine((0.4,0.5))
    # print(hb.domains)
    # print(hb.vectors)
    hb.plot_level_basis()
    plt.show()
    

if __name__ == "__main__":
    main()

    



    