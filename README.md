# **PYTHON GRAD-SHAFRANOV CutFEM SOLVER**

## *PROBLEM DESCRIPTION:*

The Grad-Shafranov equation is an nonlinear elliptic PDE which models the force balance between the plasma expansion pressure and the magnetic confinement pressure in an axisymmetrical system (for instance a tokamak device). 
Solving the equation yields the cross-section poloidal magnetic flux surfaces configuration at the equilibrium state.

While the tokamak's confining magnets' currents and positions can be adjusted to accommodate a variety of plasma pressure and current profiles, the current carried by the plasma depends on its cross-section shape, which at the same time is affected by the plasma current's self-induced magnetic field. 
This coupling implies that the problem must be solved **FREE-boundary**, for which CutFEM is suited, allowing the plasma to evolve towards the equilibrium configuration.

CutFEM is a non-conforming Finite Element Method characterised by an unfitted computational mesh, where geometries and domains are not aligned with mesh nodes but instead lie embedded, making it adapted for problems where interfaces are affected by large deformations and resizing such as free-boundary problems.
When using these unfitted methods, interfaces and domain boundaries are parametrised and monitored using level-set functions.

Additionally, the code is equiped with a **FIXED-boundary** solver, for which several analytical cases have been implemented and tested in order to validate the numering scheme. 

- **The FIXED-boundary problem** refers to an artificial case where the plasma shape is *a priori* known, and therefore the plasma region and by extension its boundary are fixed.
In this case, the external confinement magnets are irrelevant as the equilibrium state is forced on the system by fixing the plasma boundary.
Analytical solutions can be obtained by selecting the adequate source term function (plasma current model) and imposing the correct BC on the plasma boundary. 

- **The FREE-boundary problem** refers to the situation when the shape of the plasma domain is unknown.
In this case, the magnetic confinement is projected onto the computational domain's boundary using a Green's function formalism, thus providing the corresponding BC so that the plasma domain converges towards the equilibrium state iteratively. 

## *CONTENT:*

The user may find in the repository the following items:
- folder **MESHES**: contains the folders containing different example meshes. 
- folder **TESTs**: contains the test-suites, in both *.py* and *.ipynb* format, for the different standard problem cases, and the main test files in *.ipynb*
- folder **src**: contains the source code.
- file **requirements.txt**: lists the Python packages and their versions on which EQUILIPY depends.

## *CODE:*

The EQUILIPY solver is built on a CutFEM numerical scheme, where the plasma cross-section geometry is free to deform and evolve towards the equilibrium configuration. 


Adequate tolerances and maximal iteration thresholds shall be specified as inputs for both loops: internal loop, responsible of converging the poloidal magnetic field solution; external loop, responsible for converging the projected BC poloidal magnetic values.   



In the following are described the different steps in order to prepare an EQUILIPY simulation file.
The user may find already prepared examples in folder **TESTs**.

### **I. Simulation parameters**

In a first instance, the user shall provide several numerical parameters which define the problem's nature and the solver's convergence tolerances and iterative behavior. 
Among these parameters the user will find:

- **FIXED_BOUNDARY**: Enable/disable fixed plasma boundary behavior (bool).    
- **QuadratureOrder2D**: Surface numerical integration order (int).  
- **QuadratureOrder1D**: Length numerical integration order (int).  
- **ext_maxiter**: Max iterations for external loop (int).  
- **ext_tol**: Convergence tolerance for external loop (float).  
- **int_maxiter**: Max iterations for internal loop (int).  
- **int_tol**: Convergence tolerance for internal loop (float).   
- **beta**: Nitsche’s method penalty parameter (float).   
- **zeta**: Ghost penalty parameter (float).  
- **R0_axis**, **Z0_axis**: Initial guess coordinates for magnetic axis optimization (float).  
- **R0_saddle**, **Z0_saddle**: Initial guess coordinates for saddle point optimization (float).  
- **opti_maxiter**: Max iterations for critical points optimization (int).  
- **opti_tol**: Convergence tolerance for optimization (float).  

EQUILIPY can in fact solve either **FIXED-boundary** or **FREE-boundary problems**, with different order of quadratures and tolerances. 
Constraints on the arbitrary plasma boundary (cutting through the mesh) are weakly imposed using **Nitsche's method** and stabilized through **Ghost stabilisation**. 

### **II. Computational domain mesh**


### **III. Tokamak device**

After selecting an adequate computational domain mesh from folder **MESHES**, the user must provide the tokamak's geometry data and use it to declare object **Tokamak** (contained in src/Tokamak.py): 

- for the **FIXED-boundary** problem, defining a tokamak object is actually optional, however we recomment providing a mesh whose boundaries correspond to the tokamak's first wall. 
- for the **FREE-boundary**  problem, both tokamak first wall mesh and external magnets must be defined using the different available classes (see file src/Magnet.py).

### **IV. Initial plasma boundary**

The next step consists in defining the plasma cross-section initial boundary using class **InitialPlasmaBoundary** (contained in src/InitialPlasmaBoundary.py).


### **V. Initial plasma magnetic flux field (initial guess)**


### **VI. Plasma toroidal current model**


Embedded in a larger uncomforming mesh, the plasma region geometry is parametrised using a level-set function. 
Hence, the initial plasma cross-section can be arbitrarily defined by the user.



Under such circumstances, both plasma boundary and vacuum vessel wall generate cut-elements on which the FE methodology is adapted: 
- **Standard high-order approach adapted numerical integration quadratures** to integrate on each subdomain composing the cut-element
- **Nitsche's method** to weakly impose boundary conditions (BC) 
- **Ghost stabilisation** is applied to reduce irregular cut-elements instabilities

In case where the computational domain's boundary is taken as the vacuum vessel wall, the BC are still imposed weakly using Nitsche's method. 




## *EXECUTION:*

After clonning the repository with 

    $ git clone https://github.com/Elmanyer/EQUILI_PY.git
    
the code is ready is run. 

Inside the **TESTs** folder, the user may find the test-suites *TS-* files, both in *.py* and *.ipynb*, ready to execute. For python files *.py* simply execute 

    $ python TS-CASE.py

The mesh used for the simulation may be changed by commenting and uncommenting the adequate lines. These test-suites represent the simulations corresponding to the *FIXED*-boundary analytical cases, for the *LINEAR* and *NONLINEAR* plasma current models, and the *FREE*-boundary problem with *APEC* plasma current model.




