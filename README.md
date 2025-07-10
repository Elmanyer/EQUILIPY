# **PYTHON GRAD-SHAFRANOV CutFEM SOLVER**

## *PROBLEM DESCRIPTION:*

The Grad-Shafranov equation is an nonlinear elliptic PDE which models the force balance between the plasma expansion pressure and the magnetic confinement pressure in an axisymmetrical system. 
Solving the equation yields the plasma equilibrium cross-section configuration.

The problem is tackle as a free-boundary problem, where the plasma cross-section geometry is free to evolve and deform towards the equilibrium state. 
In order to deal with such configuration, the solver is based on a CutFEM numerical scheme, a non-conforming mesh Finite Element Method where the geometry is embedded in the mesh. 

Additionally, the code is equiped with a fixed-boundary solver, for which several analytical cases have been implemented and tested in order to validate the complex numering scheme. 

## *CONTENT:*

The user may find in the repository the following items:
- folder **MESHES**: contains the folders containing different example meshes. 
- folder **TESTs**: contains the test-suites, in both *.py* and *.ipynb* format, for the different standard problem cases, and the main test files in *.ipynb*
- folder **src**: contains the source code.
- file **requirements.txt**: lists the Python packages and their versions on which EQUILIPY depends.

## *CODE:*

The EQUILIPY solver is built on a CutFEM numerical scheme, where the plasma cross-section geometry is free to deform and evolve towards the equilibrium configuration. 
In a first instance, the user shall provide several numerical parameters which define the problem's nature and the solver's convergence. 

- **FIXED_BOUNDARY**: Enable/disable fixed boundary conditions (bool).  
- **GhostStabilization**: Enable/disable ghost penalty stabilization (bool).  
- **PARALLEL**: Enable/disable parallel computation (bool). (THIS OPTION IS STILL NOT IMPLEMENTED!!)  

- **plotelemsClas**: Plot element classification each iteration (bool).  
- **plotPSI**: Plot PSI solution each iteration (bool).  
- **out_proparams**: Output simulation parameters file (bool).  
- **out_elemsClas**: Output mesh element classification file (bool).  
- **out_plasmaLS**: Output plasma boundary level-set values (bool).  
- **out_plasmaBC**: Output plasma boundary condition values (bool).  
- **out_plasmaapprox**: Output plasma boundary approximation data (bool).  
- **out_ghostfaces**: Output ghost stabilization face data (bool).  
- **out_elemsys**: Output elemental matrices (bool).  
- **out_pickle**: Enable simulation data pickling (bool).  

- **dim**: Problem spatial dimension (int).  
- **QuadratureOrder2D**: Surface numerical integration order (int).  
- **QuadratureOrder1D**: Length numerical integration order (int).  
- **ext_maxiter**: Max iterations for external loop (int).  
- **ext_tol**: Convergence tolerance for external loop (float).  
- **int_maxiter**: Max iterations for internal loop (int).  
- **int_tol**: Convergence tolerance for internal loop (float).  
- **it_plasma**: Iteration to update plasma region (int).  
- **tol_saddle**: Tolerance for saddle point update (float).  
- **beta**: Nitsche’s method penalty parameter (float).  
- **Nconstrainedges**: Number of plasma boundary edges with constrained BC (int).  
- **zeta**: Ghost penalty parameter (float).  

- **PSIrelax**: Enable PSI Aitken relaxation (bool).  
- **lambda0**: Initial Aitken relaxation parameter (float).  
- **PHIrelax**: Enable PHI level-set Aitken relaxation (bool).  
- **alphaPHI**: Initial PHI Aitken relaxation parameter (float).  

- **R0_axis**, **Z0_axis**: Initial guess coordinates for magnetic axis optimization (float).  
- **R0_saddle**, **Z0_saddle**: Initial guess coordinates for saddle point optimization (float).  
- **opti_maxiter**: Max iterations for critical points optimization (int).  
- **opti_tol**: Convergence tolerance for optimization (float).  


Embedded in a larger uncomforming mesh, the plasma region geometry is parametrised using a level-set function. 
Hence, the initial plasma cross-section can be arbitrarily defined by the user.

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


## *EXECUTION:*

After clonning the repository with 

    $ git clone https://github.com/Elmanyer/EQUILI_PY.git
    
the code is ready is run. 

Inside the **TESTs** folder, the user may find the test-suites *TS-* files, both in *.py* and *.ipynb*, ready to execute. For python files *.py* simply execute 

    $ python TS-CASE.py

The mesh used for the simulation may be changed by commenting and uncommenting the adequate lines. These test-suites represent the simulations corresponding to the *FIXED*-boundary analytical cases, for the *LINEAR* and *NONLINEAR* plasma current models, and the *FREE*-boundary problem with *APEC* plasma current model.




