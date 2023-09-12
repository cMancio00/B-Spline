from __future__ import annotations
import numpy as np
from B_Spline import B_Spline
from matplotlib import pyplot as plt


class HB_Spline():
    def __init__(self,mother:B_Spline) -> None:
        self.mother = mother.compute_base()
        self.domains = [{"start":mother.knots[0],"stop":mother.knots[-1],"knots":len(mother.knots),"start_idx":0,"stop_idx":len(mother.knots)}]
        self.vectors = [{"knots":mother.knots,"level":0}]
        self.level_basis = [{"b_spline":mother,"basis":mother.get_base(),"level":0}]
        self.marked_basis = list()

    def refine(self,range:tuple):
        self.\
            add_domain(range).\
            add_vector().\
            compute_level_base().\
            mark_basis()
    
    def add_domain(self,range:tuple) -> HB_Spline:
        start_idx,stop_idx = self.find_closest_range(range)
        domain = \
        {
            "start": self.vectors[-1]["knots"][start_idx],
            "stop": self.vectors[-1]["knots"][stop_idx],
            "knots": ((self.domains[-1]["knots"]) * 2) - 1
        }
        self.update_previus_domain(start_idx,stop_idx)
        self.domains.append(domain)
        return self

    def update_previus_domain(self,start_idx,stop_idx) -> HB_Spline:
        self.domains[-1].update(
            {
            "start_idx":start_idx,
            "stop_idx":stop_idx
            }
        )
        return self

    def find_closest_range(self,range:tuple) -> tuple[int,int]:
        start_idx = (np.abs(self.vectors[-1]["knots"] - range[0])).argmin()
        stop_idx = (np.abs(self.vectors[-1]["knots"] - range[1])).argmin()
        return start_idx, stop_idx
    
    def add_vector(self) -> HB_Spline:
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
        return self

    def compute_level_base(self) -> HB_Spline:
        base = B_Spline(
            knots=self.vectors[-1]["knots"],
            order = self.mother.order
        ).compute_base()

        level_base = {
            "b_spline":base,
            "basis":base.get_base(),
            "level":self.vectors[-1]["level"]
        }
        self.level_basis.append(level_base)
        return self

    def mark_basis(self) -> HB_Spline:
        marked_basis = list()
        #Si prende l'indice di inizio e fine rifinitura
        level_domain_start = self.domains[-2]["start_idx"]
        level_domain_stop = self.domains[-2]["stop_idx"]

        #Si prende il numero di funzioni di base del livello corrente
        number_basis_function = self.level_basis[-2]["b_spline"].number_basis_function

        for i in range(number_basis_function):
            base_domain_start = self.level_basis[-2]["b_spline"].domain_range[i]["start_idx"]
            base_domain_stop = self.level_basis[-2]["b_spline"].domain_range[i]["stop_idx"]

            if (base_domain_start >= level_domain_start) and (base_domain_stop <= level_domain_stop):
                marked_basis.append(1)
            else:
                marked_basis.append(0)

        self.marked_basis.append(marked_basis)
        return self


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
    hb.refine((0.1,0.9))
    hb.refine((0.3,0.7))
    hb.refine((0.4,0.6))
    print(hb.marked_basis)

    # print(hb.domains)
    # print(hb.vectors)
    #print(hb.level_basis[0]["b_spline"].domain_range[0]["start_idx"])
    hb.plot_level_basis()
    #plt.show()
    

if __name__ == "__main__":
    main()

    



    