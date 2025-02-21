# **PYTHON GRAD-SHAFRANOV SOLVER BASED ON CUTFEM**

## *PROBLEM DESCRIPTION:*

The Grad-Shafranov equation is an nonlinear elliptic PDE which models the force balance between the plasma expansion pressure and the magnetic confinement pressure in an axisymmetrical system. 
Solving the equation yields the plasma equilibrium cross-section configuration.

The problem is tackle as a free-boundary problem, where the plasma cross-section geometry is free to evolve and deform towards the equilibrium state. 
In order to deal with such configuration, the solver is based on a CutFEM numerical scheme, a non-conforming mesh Finite Element Method where the geometry is embedded in the mesh. 

The input files *.equ.dat* and meshes available in the repository have been designed according to the ITER tokamak cross-section geometry.
In order to launch simulations with other geometries, the user shall change the geometrical parameters for the vacuum vessel and the external coils and solenoids from the input file *.equ.dat*.

## *CODE:*

The EQUILIPY_CutFEM solver is built on a CutFEM numerical scheme, where the plasma cross-section geometry is free to deform and evolve towards the equilibrium configuration. 
Both the plasma region and the tokamak's vacuum vessel wall geometries are parametrised using two distinct level-set functions, both domains being embedded in a larger uncomforming mesh.
Hence, both the initial plasma cross-section and the fixed vacuum vessel wall geometries can be arbitrarily defined as inputs by the user.

Under such circumstances, both plasma boundary and vacuum vessel wall generate cut-elements on which the FE methodology is adapted: 
- **Standard high-order approach adapted numerical integration quadratures** to integrate on each subdomain composing the cut-element
- **Nitsche's method** to weakly impose boundary conditions (BC) 
- **Ghost stabilisation** is applied to reduce irregular cut-elements instabilities

In case where the computational domain's boundary is taken as the vacuum vessel wall, the BC are still imposed weakly using Nitsche's method. 

EQUILIPY_CutFEM can solve two distinct problems: either *fixed-boundary* or *free-boundary* problems:
- **The fixed-boundary problem** refers to an artificial case where the plasma shape is *a priori* known, and therefore the plasma region and by extension its boundary are fixed.
In this case, both the vacuum vessel wall parametrisation and the external magnets configuration are irrelevant, as the equilibrium state is forced on the system by fixing the plasma boundary.
Analytical solutions can be obtained by selecting the adequate source term function (plasma current model) and imposing the correct BC on the plasma boundary. 
- **The free-boundary problem** refers to the situation when the shape of the plasma domain is unknown.
In this case, the magnetic confinement is projected onto the vacuum vessel wall using a Green's function formalism, thus providing the corresponding BC so that the plasma domain converges towards the equilibrium state iteratively. 
Adequate tolerances and maximal iteration thresholds shall be specified as inputs for both loops: internal loop, responsible of converging the poloidal magnetic field solution; external loop, responsible for converging the projected BC poloidal magnetic values.   

## *CONTENT:*
- folder **src**: contains the source code
- folder **CASES**: contains the input files *.equ.dat* for the different problem cases
- folder **MESHES**: contains the files describing different meshes
- folder **TESTs**: contains the test-suites, in both *.py* and *.ipynb* format, for the different standard problem cases, and the main test files in *.ipynb*

## *EXECUTION:*

After clonning the repository with 

    $ git clone https://github.com/Elmanyer/EQUILI_PY.git
    
the code is ready is run. 

Inside the **TESTs** folder, the user may find the test-suites *TS-* files, both in *.py* and *.ipynb*, ready to execute. For python files *.py* simply execute 

    $ python TS-CASE.py

The mesh used for the simulation may be changed by commenting and uncommenting the adequate lines. These test-suites represent the simulations corresponding to the *FIXED*-boundary analytical cases, for the *LINEAR*, *NONLINEAR* and *ZHENG* plasma current models, and the *FREE*-boundary problem with *PROFILES* plasma current model.

To launch other simulations, the user may use testing files *MainTestEquilipyFIXED.ipynb* and *MainTestEquilipyFREE.ipynb*, where all available meshes have been included. For fixed-boundary problem simulations, meshes can be adjusted to the fixed plasma cross-section (*-REDUCED* meshes); on the other hand, for free-boundary problem simulations larger meshes should be used, preparing for plasma cross-ection deformations. 

## *INPUT FILE*

Input files *.equ.dat* in folder **CASES** contain the simulation parameters for different configurations, defining the problem case, the different parametrised geometries and the numerical treatment. The parameters are organised according to the following blocks:

- **PHYSICAL PROBLEM:**
    - **PROBLEM CASE PARAMETERS:** parameters controlling the simulation problem case (*PLASB*, *PLASG*, *PLASC*, *VACVE*).
    - **VACUUM VESSEL GEOMETRY:** dimensionless parameters characterising the tokamak's vacuum vessel cross-section.
    - **PLASMA REGION GEOMETRY:** initial/free plasma cross-section parameters for the parametrised case (*PLASG* = PARAM).   
    - **PARAMETERS FOR PRESSURE AND TOROIDAL FUNCTION PROFILES**: *PROFILES* plasma current model parameters.
    - **PARAMETERS FOR EXTERNAL COILS AND SOLENOIDS:** tokamak's external magnets positions and currents. 

- **NUMERICAL TREATMENT:** parameters involved in the numerical scheme (tolerances, maximum iterations...) and numerical processes (initial guess, tolerances...).

