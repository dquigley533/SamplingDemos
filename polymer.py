import numpy as np
import math as m

import numpy as np
#from numba import jitclass          # import the decorator
#from numba import int32, float32    # import the types

#spec = [        
#    ('Ndims',  int32),
#    ('Nbeads', int32),       
#]

#@jitclass(spec)
class polymer:
    """A class which defines a simple bead-spring model of a freely jointed 
       polymer chain."""

    Ndims  = 3
    Nbeads =  10

    r0 = 0.7  # Equilibrium bead-bead bond distance
    K  = 40   # FENE spring constant
    R  = 0.3  # Maximum deviation from r0

    epsilon = 1.0                   # Lennard-Jones energy parameter
    sigma   = r0/(2.0**(1.0/6.0))   # Lennard-Jones sigma  parameter
    rc      = 2.5*sigma             # Lennard-Jones cutoff

    def __init__(self, Ndims, Nbeads):
        "Constructor for class. Takes number of dimensions and number of beads as input"
  
        self.Ndims  = Ndims    # Number of dimensions
        self.Nbeads = Nbeads   # Set number of beads
       
        self.rpos = np.zeros((Nbeads, Ndims)) # Create array of positions

        # Initialise as a linear chain along the first dimension with equilibrium bond length
        xpos = 0.0
        for pos in self.rpos:
            pos[0,] = xpos
            xpos += self.r0

        # Precompute boring constants that don't change
        self.sigma6  = self.sigma**6
        self.sigma12 = self.sigma6**2
        ir6 = 1.0/self.rc**6
        ir12 = ir6*ir6
        self.ljshift   = 4*self.epsilon*( ir12*self.sigma12 - ir6*self.sigma6 )

        self.invR = 1.0/self.R
        self.fenefactor = -0.5*self.K*self.R**2

        self.total_energy = self.energy() 


    def LJ_nb(self,r):
        "Compute truncated and shifted Lennard-Jones potential  "

        if r < self.rc:
            ir6 = 1.0/r**6
            ir12 = ir6*ir6
            return 4*self.epsilon*( ir12*self.sigma12 - ir6*self.sigma6 ) - self.ljshift
        else:
            return 0.0

    def FENE(self,r):
        "Compute FENE bond stretch potential"    

        arg = 1 - ((r-self.r0)*self.invR)**2
        if arg <= 0.0:
            return 1E10 #np.finfo(np.float64).max
        else:
            return self.fenefactor*m.log(arg)

    def energy(self):
        "Compute total energy of the polymer chain"   

        # Bonded energy from FENE springs
        bond_energy = 0.0
        for ibead in range(0,self.Nbeads - 1):
            rvect = self.rpos[ibead+1] - self.rpos[ibead]
            rmag  = np.linalg.norm(rvect)
            bond_energy = bond_energy + self.FENE(rmag)

        # Non-bonded energy from LJ interactions
        nb_energy = 0.0
        for ibead in range(0,self.Nbeads-2):
            for jbead in range(ibead+2,self.Nbeads): 
                rvect = self.rpos[ibead] - self.rpos[jbead]
                rmag  = np.linalg.norm(rvect)

                nb_energy = nb_energy + self.LJ_nb(rmag)

        self.total_energy = nb_energy + bond_energy


        return self.total_energy


    def local_energy(self,ibead):
        "Computes all contributions to the energy involving ibead"

        # Bonded energy from FENE springs
        bond_energy = 0.0
        
        if (ibead < self.Nbeads-1):
            rvect = self.rpos[ibead+1] - self.rpos[ibead]
            rmag  = np.linalg.norm(rvect)
            bond_energy = bond_energy + self.FENE(rmag)

        if (ibead > 0):
            rvect = self.rpos[ibead-1] - self.rpos[ibead]
            rmag  = np.linalg.norm(rvect)
            bond_energy = bond_energy + self.FENE(rmag)

        # Non-bonded energy from LJ interactions
        nb_energy = 0.0
        for jbead in range(0, ibead-1):
            rvect = self.rpos[ibead] - self.rpos[jbead]
            rmag  = np.linalg.norm(rvect)
            nb_energy = nb_energy + self.LJ_nb(rmag)

        for jbead in range(ibead+2, self.Nbeads):
            rvect = self.rpos[ibead] - self.rpos[jbead]
            rmag  = np.linalg.norm(rvect)
            nb_energy = nb_energy + self.LJ_nb(rmag)

        return bond_energy + nb_energy

    def end2end(self):
        "End to end distance of the polymer chain"

        rvect = self.rpos[self.Nbeads-1] - self.rpos[0]
        return np.linalg.norm(rvect)
