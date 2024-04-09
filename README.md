# SamplingDemos

Collection of notebooks demonstrating various multicanonical sampling strategies on a simple bead-spring single chain polymer.

 - Basic_MMC.ipynb : Simple canonical (single temperature) MC
 - MUCA.ipynb      : Multicanonical sampling with iteratively refined weights
 - WL.ipynb        : Wang-Landau method based on density of states in energy
 - TMMC.ipynb      : Transition matrix MC method

Requirements are Jupyter, Numpy and Matplotlib.

If using the SCRTP managed Linux computers then:

```
module load GCC/11.3.0 OpenMPI/4.1.4 SciPy-bundle/2022.05 matplotlib/3.5.2 IPython/8.5.0
```

will get you everything you need. See:

https://docs.scrtp.warwick.ac.uk/taskfarm-pages/applications-pages/jupyter.html

for how to launch a remotely accessible Jupyter server on a shared SCRTP server.