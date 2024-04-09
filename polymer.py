import numpy as np
import math as m

class polymer:
    """A class which defines a simple bead-spring model of a freely jointed 
       polymer chain."""

    Ndims  = 3    # Number of dimensions 
    Nbeads =  10  # Number of beads

    _r0 = 0.7  # Equilibrium bead-bead bond distance
    _K  = 40   # FENE spring constant
    _R  = 0.3  # Maximum deviation from r0

    _epsilon = 1.0                    # Lennard-Jones energy parameter
    _sigma   = _r0/(2.0**(1.0/6.0))   # Lennard-Jones sigma  parameter
    _rc      = 2.5*_sigma             # Lennard-Jones cutoff

    def __init__(self, Ndims, Nbeads):
        "Constructor for class. Takes number of dimensions and number of beads as input"
  
        self.Ndims  = Ndims    # Number of dimensions
        self.Nbeads = Nbeads   # Set number of beads
       
        self.rpos = np.zeros((Nbeads, Ndims)) # Create array of positions

        # Initialise as a linear chain along the first dimension with equilibrium bond length
        xpos = 0.0
        for pos in self.rpos:
            pos[0,] = xpos
            xpos += self._r0

        # Precompute boring constants that don't change
        self._sigma6  = self._sigma**6
        self._sigma12 = self._sigma6**2
        ir6 = 1.0/self._rc**6
        ir12 = ir6*ir6
        self._ljshift   = 4*self._epsilon*( ir12*self._sigma12 - ir6*self._sigma6 )

        self._invR = 1.0/self._R
        self._fenefactor = -0.5*self._K*self._R**2

        self.total_energy = self.energy() 


    def LJ_nb(self,r):
        "Compute truncated and shifted Lennard-Jones potential  "

        if r < self._rc:
            ir6 = 1.0/r**6
            ir12 = ir6*ir6
            return 4*self._epsilon*( ir12*self._sigma12 - ir6*self._sigma6 ) - self._ljshift
        else:
            return 0.0

    def FENE(self,r):
        "Compute FENE bond stretch potential"    

        arg = 1 - ((r-self._r0)*self._invR)**2
        if arg <= 0.0:
            return 1E10 #np.finfo(np.float64).max
        else:
            return self._fenefactor*m.log(arg)

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
