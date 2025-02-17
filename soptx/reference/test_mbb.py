from dolfin import SubDomain


# Boundary Condtion
class SimDB1(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0)

