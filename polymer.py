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
    #_rc      = 2.5*_sigma            # Lennard-Jones cutoff

    def __init__(self, Ndims, Nbeads):
        "Constructor for class. Takes number of dimensions and number of beads as input"
  
        self.Ndims  = Ndims    # Number of dimensions
        self.Nbeads = Nbeads   # Set number of beads
       
        self.rpos = np.zeros((Nbeads, Ndims))   # Create array of positions
        self.vels = np.zeros((Nbeads, Ndims))   # Create array of velocities
        self.forces = np.zeros((Nbeads, Ndims)) # Create array of forces

        # Initialise as a linear chain along the first dimension with equilibrium bond length
        xpos = 0.0
        for pos in self.rpos:
            pos[0,] = xpos
            xpos += self._r0

        # Precompute boring constants that don't change
        self._sigma6  = self._sigma**6
        self._sigma12 = self._sigma6**2
        #ir6 = 1.0/self._rc**6
        #ir12 = ir6*ir6
        #self._ljshift   = 4*self._epsilon*( ir12*self._sigma12 - ir6*self._sigma6 )

        self._ljrmin = self._sigma*2.0**(1.0/6.0)

        self._invR = 1.0/self._R
        self._fenefactor = -0.5*self._K*self._R**2

        self.total_energy = self.energy() 
        self.compute_forces()


    def LJ_nb(self,r):
        "Compute Lennard-Jones potential for non-bonded interactions"

        #if r < self._rc:
        ir6 = 1.0/r**6
        ir12 = ir6*ir6
        return 4*self._epsilon*( ir12*self._sigma12 - ir6*self._sigma6 ) #- self._ljshift
        #else:
        #    return 0.0

    def LJ_nb_fast(self,rsq):
        "Compute Lennard-Jones potential for non-bonded interactions given r squared"

        #if r < self._rc:
        ir6 = 1.0/rsq**3
        ir12 = ir6*ir6
        return 4*self._epsilon*( ir12*self._sigma12 - ir6*self._sigma6 ) #- self._ljshift

    def FENE(self,r):
        "Compute FENE bond stretch potential"    

        arg = 1 - ((r-self._r0)*self._invR)**2
        if arg <= 0.0:
            return 1E10 #np.finfo(np.float64).max
        else:
            return self._fenefactor*np.log(arg)

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
                #rmag  = np.linalg.norm(rvect)
                rsq   = np.dot(rvect,rvect)
                nb_energy = nb_energy + self.LJ_nb_fast(rsq)

        self.total_energy = nb_energy + bond_energy


        return self.total_energy

    def local_energy(self,ibead):
        "Computes all contributions to the energy involving ibead"

        # Bonded energy from FENE springs
        bond_energy = 0.0

        #sqrtfunc = np.linalg.norm
        #dotfunc  = np.dot
        
        if (ibead < self.Nbeads-1):
            rvect = self.rpos[ibead+1] - self.rpos[ibead]
            rmag  = np.sqrt(np.dot(rvect,rvect))
            bond_energy = bond_energy + self.FENE(rmag)

        if (ibead > 0):
            rvect = self.rpos[ibead-1] - self.rpos[ibead]
            rmag  = np.sqrt(np.dot(rvect,rvect))
            bond_energy = bond_energy + self.FENE(rmag)

        # Non-bonded energy from LJ interactions
        nb_energy = 0.0
        for jbead in range(0, ibead-1):
            rvect = self.rpos[ibead] - self.rpos[jbead]
            #rmag  = normfunc(rvect)
            rsq   = np.dot(rvect,rvect)
            nb_energy = nb_energy + self.LJ_nb_fast(rsq)

        for jbead in range(ibead+2, self.Nbeads):
            rvect = self.rpos[ibead] - self.rpos[jbead]
            #rmag  = normfunc(rvect)
            rsq   = np.dot(rvect,rvect)
            nb_energy = nb_energy + self.LJ_nb_fast(rsq)

        return bond_energy + nb_energy

    def force_on_bead(self, ibead):
        """Compute total force vector acting on bead `ibead`.

        Returns a numpy array of length `Ndims` with the force contributions
        from bonded FENE springs (neighbours ibead-1 and ibead+1) and
        non-bonded Lennard-Jones interactions with all other beads excluding
        immediate bonded neighbours.

        Edge cases:
        - If a pair distance r is zero, that pair is skipped to avoid
          division-by-zero. Very small denominators in the FENE expression
          are clipped to a small value to avoid numerical overflow.
        """
        # initialize force vector
        f = np.zeros(self.Ndims)

        # small cutoff to prevent division by zero
        tiny = 1e-12

        # Bonded neighbours (FENE)
        if ibead < self.Nbeads - 1:
            # neighbour i+1
            rvect = self.rpos[ibead] - self.rpos[ibead+1]
            r = np.linalg.norm(rvect)
            if r > tiny:
                arg = 1 - ((r - self._r0) * self._invR)**2
                # avoid division by zero / negative argument
                denom = arg if arg > tiny else tiny
                # dU/dr for FENE = K*(r - r0)/arg  -> force = -dU/dr * r_hat
                fmag = - self._K * (r - self._r0) / denom
                f += fmag * (rvect / r)

        if ibead > 0:
            # neighbour i-1
            rvect = self.rpos[ibead] - self.rpos[ibead-1]
            r = np.linalg.norm(rvect)
            if r > tiny:
                arg = 1 - ((r - self._r0) * self._invR)**2
                denom = arg if arg > tiny else tiny
                fmag = - self._K * (r - self._r0) / denom
                f += fmag * (rvect / r)

        # Non-bonded LJ interactions (exclude immediate bonded neighbours)
        # j in [0, ibead-2] and [ibead+2, Nbeads-1]
        for jbead in range(0, ibead-1):
            rvect = self.rpos[ibead] - self.rpos[jbead]
            r = np.linalg.norm(rvect)
            if r > tiny:
                ir6 = 1.0 / r**6
                ir12 = ir6 * ir6
                # dU/dr for LJ (with sigma factors included)
                dUdr = 4.0 * self._epsilon * ( -12.0 * ir12 * self._sigma12 / r
                                                + 6.0 * ir6 * self._sigma6 / r )
                # force is -dU/dr * r_hat
                f += -dUdr * (rvect / r)

        for jbead in range(ibead+2, self.Nbeads):
            rvect = self.rpos[ibead] - self.rpos[jbead]
            r = np.linalg.norm(rvect)
            if r > tiny:
                ir6 = 1.0 / r**6
                ir12 = ir6 * ir6
                dUdr = 4.0 * self._epsilon * ( -12.0 * ir12 * self._sigma12 / r
                                                + 6.0 * ir6 * self._sigma6 / r )
                f += -dUdr * (rvect / r)

        return f

    def compute_forces(self):
        """Compute forces for all beads and return an array of shape (Nbeads, Ndims).

        Vectorized O(N^2) implementation that computes pairwise displacement
        arrays and accumulates bonded (FENE) and non-bonded (LJ) forces.
        """
        pos = self.rpos  # (N, D)
        N = self.Nbeads
        allf = np.zeros((N, self.Ndims))

        # pairwise displacement: r_i - r_j (N, N, D)
        rij = pos[:, None, :] - pos[None, :, :]
        r = np.linalg.norm(rij, axis=2)  # (N, N)
        tiny = 1e-12

        # Non-bonded LJ interactions: pairs with |i-j| >= 2
        idx = np.arange(N)
        diff = np.abs(idx[:, None] - idx[None, :])
        mask_nb = diff >= 2

        iu, ju = np.where(np.triu(mask_nb, k=1))
        if iu.size > 0:
            rv = rij[iu, ju, :]
            rpair = np.linalg.norm(rv, axis=1)
            valid = rpair > tiny
            if np.any(valid):
                rvv = rv[valid]
                rvec = rpair[valid]
                ir6 = 1.0 / (rvec**6)
                ir12 = ir6 * ir6
                # dU/dr for LJ: 4*epsilon*( -12*sigma12/r^13 + 6*sigma6/r^7 )
                dUdr = 4.0 * self._epsilon * ( -12.0 * ir12 * self._sigma12 / rvec
                                               + 6.0 * ir6 * self._sigma6 / rvec )
                # force on i from j: -dU/dr * r_hat
                fij = (-dUdr)[:, None] * (rvv / rvec[:, None])
                # accumulate forces (i gets +fij, j gets -fij)
                valid_idx_i = iu[valid]
                valid_idx_j = ju[valid]
                np.add.at(allf, valid_idx_i, fij)
                np.add.at(allf, valid_idx_j, -fij)

        # Bonded FENE interactions: only neighbouring pairs (i, i+1)
        if N > 1:
            iu = np.arange(0, N-1)
            ju = iu + 1
            rv = rij[iu, ju, :]
            rpair = np.linalg.norm(rv, axis=1)
            valid = rpair > tiny
            if np.any(valid):
                rvv = rv[valid]
                rvec = rpair[valid]
                arg = 1 - ((rvec - self._r0) * self._invR)**2
                denom = np.where(arg > tiny, arg, tiny)
                # dU/dr for FENE = -K*(r - r0)/arg  -> force = -dU/dr * r_hat
                fmag = - self._K * (rvec - self._r0) / denom
                fij = fmag[:, None] * (rvv / rvec[:, None])
                valid_idx_i = iu[valid]
                valid_idx_j = ju[valid]
                np.add.at(allf, valid_idx_i, fij)
                np.add.at(allf, valid_idx_j, -fij)

        self.forces = allf
        return 

    def end2end(self):
        "End to end distance of the polymer chain"

        rvect = self.rpos[self.Nbeads-1] - self.rpos[0]
        return np.linalg.norm(rvect)

    def kinetic_energy(self):
        "Compute total kinetic energy of the polymer chain"

        ke = 0.0
        for ibead in range(0,self.Nbeads):
            vmag2 = np.dot(self.vels[ibead], self.vels[ibead])
            ke = ke + 0.5*vmag2

        return ke
    
    def lj6(self, r):
        "LJ6 potential energy function"
        
        ir6 = 1.0/r**6
        U =  4*self._epsilon*( - ir6*self._sigma6 ) 

        return U
    
    def lj12(self, r):
        "LJ12 potential energy function"
        
        ir12 = 1.0/r**12
        U =  4*self._epsilon*( ir12*self._sigma12 ) 

        return U


    def lj6_inv(self, U):
        "Inverse function for LJ6 potential energy"
        
        # If U = 4 epsilon ( - sigma6/r6 ), then
        # r6 = - 4 epsilon sigma6 / U
        r6 = - 4 * self._epsilon * self._sigma6 / U
        r = r6**(1.0/6.0)

        return r

    def lj12_inv(self, U):
        "Inverse function for LJ12 potential energy"
        
        # If U = 4 epsilon ( sigma12/r12 ), then
        # r12 = 4 epsilon sigma12 / U
        r12 = 4 * self._epsilon * self._sigma12 / U
        r = r12**(1.0/12.0)

        return r
    

    def lj_inv(self, U, sign):
        "Inverse function for LJ potential energy"
        
        # If U = 4 epsilon ( sigma12/r12 - sigma6/r6 ), then
        # Rearranging gives a quadratic in r6:
        # (4 epsilon sigma12) - (U r6) - (4 epsilon sigma6 r6^2) = 0
        # Solving using quadratic formula:

        # Sanity check for U
        if (U > 0.0) and (sign < 0.0):
            return np.inf # Can never reach positive U if moving apart
 
        if sign <= 0.0:
            num = -1 - np.sqrt(1+U/self._epsilon)
        else:
            num = -1 + np.sqrt(1+U/self._epsilon)

        denom = 0.5*U/self._epsilon

        r = self._sigma * (num/denom)**(1.0/6.0)

        return r
    
    def lj_time(self, imove, isource, vel, max_delta_U):
        '''Compute time to collision for the full LJ potential'''

        # Vector from moving bead to source bead
        rsep = self.rpos[isource] - self.rpos[imove]

        # Distance squared between moved bead and source bead
        rsq = np.dot(rsep, rsep)
        r  = np.sqrt(rsq)

        # Unit vector along rsep
        rhat = rsep / r

        # Initialise distance we can move along the "bond"
        dist = 0.0

        # If there's positive projection of rsep onto vel, 
        # the "bond" is getting shorter.
        vdot = np.dot(rsep, vel)  
        if (vdot >= 0.0):
            # If the "bond" is currently stretched beyond ljrmin,
            # we can move at least (r-ljrmin) along the "bond" and
            # restart calculation from ljrmin.
            if (r-self._ljrmin) > 0.0:
                #print("Moving to ljrmin from beyond minimum")
                dist += r-self._ljrmin
                rsep = self._ljrmin * rhat
                r = self._ljrmin
                rsq = r*r

        # Otherwise the "bond" is getting longer
        else:
            # If the "bond" is currently compressed below ljrmin,
            # we can move at least (rljrmin-r) along the bond and
            # restart calculation from ljrmin.
            if (self._ljrmin - r) > 0.0:
                #print("Moving to ljrmin from within minimum")
                dist += self._ljrmin - r
                rsep = self._ljrmin * rhat
                r = self._ljrmin
                rsq = r*r

        #print(f"After adjusting r = {r}, dist = {dist}")

        # Current LJ energy (will be -epsilon if we've moved to minimum)
        U0 = self.LJ_nb(r)

        # Separation at which collision would occur
        rcoll = self.lj_inv(U0 + max_delta_U, sign=np.sign(vdot))

        #print(f"rcoll = {rcoll}")

        # Add distance along rsep vector to collision point
        dist += abs(rcoll - r)

        #print(f"Total dist to collision = {dist}")

        # So now we have (vel*tcoll) dot rhat = dist
        # and time to collision is
        tcoll = dist / abs(np.dot(vel, rhat))

        return tcoll        

    
    def fene_inv(self, U, sign):
        "Inverse function for FENE potential energy"
        
        # If U = -0.5 K R^2 ln( 1 - ((r - r0)/R)^2 ), then
        # exp( -2 U / (K R^2) ) = 1 - ((r - r0)/R)^2
        # ((r - r0)/R)^2 = 1 - exp( -2 U / (K R^2) )
        # (r - r0)/R = +/- sqrt( 1 - exp( -2 U / (K R^2) ) )
        # r = r0 +/- R * sqrt( 1 - exp( -2 U / (K R^2) ) )
        
        expterm = np.exp( -2.0 * U / (self._K * self._R**2) )
        sqrtterm = np.sqrt( 1.0 - expterm )

        if sign <= 0.0:
            r = self._r0 - self._R * sqrtterm
        else:
            r = self._r0 + self._R * sqrtterm

        return r
        
    def fene_time(self, imove, isource, vel, max_delta_U):
        '''Compute time to collision for the FENE potential'''

        # Vector from moving bead to source bead
        rsep = self.rpos[isource] - self.rpos[imove]

        # Distance squared between moved bead and source bead
        rsq = np.dot(rsep, rsep)
        r  = np.sqrt(rsq)

        # Unit vector along rsep
        rhat = rsep / r

        # Initialise distance we can move along the bond
        dist = 0.0

        # If there's positive projection of rsep onto vel, 
        # the bond is getting shorter.
        vdot = np.dot(rsep, vel)  
        if (vdot >= 0.0):
            # If the bond is currently stretched beyond r0,
            # we can move at least (r-r0) along the bond and
            # restart calculation from r0.
            if (r-self._r0) > 0.0:
                dist += r-self._r0
                rsep = self._r0 * rhat
                r = self._r0
                rsq = r*r

        # Otherwise the bond is getting longer
        else:
            # If the bond is currently compressed below r0,
            # we can move at least (r0-r) along the bond and
            # restart calculation from r0.
            if (self._r0 - r) > 0.0:
                dist += self._r0 - r
                rsep = self._r0 * rhat
                r = self._r0
                rsq = r*r

        #print(f"After adjusting r = {r}, dist = {dist}")

        # Current FENE energy (will be zero if we've moved to r0 already)
        U0 = self.FENE(r)

        # Separation at which collision would occur
        rcoll = self.fene_inv(U0 + max_delta_U, sign=-1*np.sign(vdot))

        #print(f"rcoll = {rcoll}")

        # Add distance along rsep vector to collision point
        dist += abs(rcoll - r)

        #print(f"Total dist to collision = {dist}")

        # So now we have (vel*tcoll) dot rhat = dist
        # and time to collision is
        tcoll = dist / abs(np.dot(vel, rhat))

        return tcoll
        

    def lj6_time(self, imove, isource, vel, max_delta_U):
        '''Compute time to collision for the attractive LJ6 potential'''

        # Vector from moving bead to source bead
        rsep = self.rpos[isource] - self.rpos[imove]

        # If there's positive projection of sep onto vel, return NULL
        # as imove is moving toward source bead or perpendicular
        vdot = np.dot(rsep, vel)  
        if (vdot >= 0.0):
            return None

        # Distance squared between moved bead and source bead
        rsq = np.dot(rsep, rsep)
        r  = np.sqrt(rsq)

        # Unit vector along rsep
        rhat = rsep / r

        # Current lj6 energy
        U0 = self.lj6(r)

        # Separation at which collision would occur        
        rcoll = self.lj6_inv(U0 + max_delta_U)

        # Distance along rsep vector to collision point
        dist = rcoll - r

        # So now we have (vel*tcoll) dot rhat = dist
        # and time to collision is
        tcoll = dist / (np.dot(vel, rhat))

        return tcoll
    
    def lj12_time(self, imove, isource, vel, max_delta_U):
        '''Compute time to collision for the repulsive LJ12 potential'''

        # Vector from moving bead to source bead
        rsep = self.rpos[isource] - self.rpos[imove]

        # If there's negative projection of sep onto vel, return NULL
        # as imove is moving away from source bead or perpendicular
        vdot = np.dot(rsep, vel)  
        if (vdot <= 0.0):
            return None

        # Distance squared between moved bead and source bead
        rsq = np.dot(rsep, rsep)
        r = np.sqrt(rsq)

        # Unit vector along rsep
        rhat = rsep / r

        # Current lj12 energy
        U0 = self.lj12(r)

        # Separation at which collision would occur        
        rcoll = self.lj12_inv(U0 + max_delta_U)

        # Distance along rsep vector to collision point
        dist = r - rcoll

        # So now we have (vel*tcoll) dot rhat = dist
        # and time to collision is
        tcoll = dist / (np.dot(vel, rhat))

        return tcoll