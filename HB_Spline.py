import numpy as np
from B_Spline import B_Spline


class HB_Spline():
    def __init__(self,mother:B_Spline) -> None:
        self.mother = mother.compute_base()
        self.domains = [{"start":mother.knots[0],"stop":mother.knots[-1],"knots":len(mother.knots)}]
        self.vector = [mother.knots]
        self.level_basis = []

    def refine(self,range:tuple):
        self.add_domain(range)
    
    def add_domain(self,range:tuple):
        start_idx,stop_idx = self.find_closest_range(range)
        domain = \
        {
            "start": self.vector[-1][start_idx],
            "stop": self.vector[-1][stop_idx],
            "knots": ((self.domains[-1]["knots"]) * 2) - 1
        }
        self.domains.append(domain)

    def find_closest_range(self,range:tuple):
        start_idx = (np.abs(self.vector[-1] - range[0])).argmin()
        stop_idx = (np.abs(self.vector[-1] - range[1])).argmin()
        return start_idx, stop_idx

        return start,stop
    
        

    


def main():
    T = np.linspace(0, 1, 7 + 1)
    mother = B_Spline(T, 3)
    hb = HB_Spline(mother)
    hb.refine((0.2,0.8))
    hb.refine((0.4,0.5))
    print(hb.domains)

if __name__ == "__main__":
    main()

    



    