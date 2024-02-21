## Instructions to set up a working environment

### Below settings by Hui, working on Windows:

#### Using Anaconda to install every package.

1. Download Anaconda

2. Open Anaconda Prompt in Windows searching

3. > conda create -n geo 

4. > conda activate geo

5. > conda install mayavi traits traitsui qt pyqt vtk scipy spyder 

6. > conda install -c haasad pypardiso

7. Open Anaconda, under geo environment open Spyder

### files instruction: 

1. files in geometrylab folder created by Davide, nothing need to be changed.

2. if want to test how it works, try python files in geometrylab/test: ex. run paneling.py, then a GUI window will be opened.

3. if want to add a new project, create a new folder named 'archgeolab'. the mesh geometry, optimization and GUI will be based on the files in geometrylab.

4. archgeolab/archgeometry: meshpy.py --> quadrings.py --> gridshell.py --> gui_basic.py --> project folder (proj_orthonet)

5. archgeolab/proj_orthonet: guidedprojection_orthonet.py --> opt_gui_orthonet.py --> readfile_orthonet.py

   


-------------------------
### Below setting by Victor:

We need to use conda for pypardiso.

> conda create -n geometrylab python=3.6

> conda activate geometrylab

> pip install numpy scipy

> pip install python-vtk

> pip install mayavi --no-cache

> conda install -c haasad pypardiso

> conda install pyface

