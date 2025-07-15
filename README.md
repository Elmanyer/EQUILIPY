# **PYTHON GRAD-SHAFRANOV CutFEM SOLVER**

## *PROBLEM DESCRIPTION:*

The Grad-Shafranov (GS) equation is an nonlinear elliptic PDE which models the force balance between the plasma expansion pressure and the magnetic confinement pressure in an axisymmetrical system (for instance a tokamak device). 
Solving the equation yields the cross-section poloidal magnetic flux surfaces configuration at the equilibrium state.

While the tokamak's confining magnets' currents and positions can be adjusted to accommodate a variety of plasma pressure and current profiles, the current carried by the plasma depends on its cross-section shape, which at the same time is affected by the plasma current's self-induced magnetic field. 
This coupling implies that the problem must be solved **FREE-boundary**, for which CutFEM is suited, allowing the plasma to evolve towards the equilibrium configuration.

CutFEM is a non-conforming Finite Element Method characterised by an unfitted computational mesh, where geometries and domains are not aligned with mesh nodes but instead lie embedded, making it adapted for problems where interfaces are affected by large deformations and resizing such as free-boundary problems.
When using these unfitted methods, interfaces and domain boundaries are parametrised and monitored using level-set functions.

Additionally, the code is equiped with a **FIXED-boundary** solver, for which several analytical cases have been implemented and tested in order to validate the numering scheme. 

- **The FIXED-boundary problem** refers to an artificial case where the plasma shape is *a priori* known, and therefore the plasma region and by extension its boundary are fixed.
In this case, the external confinement magnets are irrelevant as the equilibrium state is forced on the system by fixing the plasma boundary.
Analytical solutions can be obtained by selecting the adequate source term function (plasma current model) and imposing the correct boundary conditions (BC) on the plasma boundary. 

- **The FREE-boundary problem** refers to the situation when the shape of the plasma domain is unknown.
In this case, the magnetic confinement is projected onto the computational domain's boundary using a Green's function formalism, thus providing the corresponding BC so that the plasma domain converges towards the equilibrium state iteratively. 

## *INSTALLATION:*

After clonning the repository with 

    $ git clone https://github.com/Elmanyer/EQUILI_PY.git
    
install all required packages inside the desired python environment or directly in the local machine using *pip* with command line

    $ pip install -r requirements.txt

The code is ready is run. 

Inside the **TESTs** folder, the user may find the test-suites *TS-* files, both in *.py* and *.ipynb*, ready to execute. For python files *.py* simply execute 

    $ python TS-CASE.py

## *CONTENT:*

The user may find in the repository the following items:
- folder **MESHES**: contains the folders containing different example meshes. 
- folder **TESTs**: contains the test-suites, in both *.py* and *.ipynb* format, for the different standard problem cases, and the main test files in *.ipynb*
- folder **src**: contains the source code.
- file **requirements.txt**: lists the Python packages and their versions necessary to run EQUILIPY.

## *CODE:*

The EQUILIPY solver is built on a CutFEM numerical scheme, where the plasma cross-section geometry is free to deform and evolve towards the equilibrium configuration.
The general solution strategy for solving FREE-boundary problems involves an iterative approach based on a **double-loop structure**: 

- *in the external loop*, the algorithm looks for the **convergence of the boundary values**, projected onto the computational domain's boundary using a Green's function formalism.
- *in the internal loop*, using as BC the values obtained in the external loop, the algorithm **solves iteratively the GS free-boundary Boundary Value Problem (BVP)** until convergence.

Adequate tolerances and maximal iteration thresholds shall be specified as inputs for both loops: internal loop, responsible of converging the poloidal magnetic field solution; external loop, responsible for converging the projected BC poloidal magnetic values.  

## *EXECUTION:*

In the following are described the different steps in order to prepare an EQUILIPY simulation file.
The user may find already prepared examples in folder **TESTs**.

### **I. Simulation parameters**

In a first instance, the user shall provide several **numerical parameters** which define the problem's nature and the solver's convergence tolerances and iterative behavior. 
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
- **R0_axis**, **Z0_axis**: Initial guess coordinates for magnetic axis nonlinear solver (float).  
- **R0_saddle**, **Z0_saddle**: Initial guess coordinates for saddle point nonlinear solver (float).  
- **opti_maxiter**: Max iterations for critical points nonlinear solver (int).  
- **opti_tol**: Convergence tolerance for nonlinear solver (float).  

EQUILIPY can solve either **FIXED-boundary** or **FREE-boundary problems** with different order of quadratures and tolerances. 
Constraints on the arbitrary plasma boundary (cutting through the mesh) are weakly imposed using **Nitsche's method** and stabilized through **Ghost stabilisation**. 
Poloidal magnetic flux critical points used for the solution's normalisation are searched using nonlinear solver (optimization algorithms). 

After defining the equilibrium problem using 

    $ Equilibrium = GradShafranovSolver()

declare the different parameters, for instance

    $ Equilibrium.FIXED_BOUNDARY = False
    $ Equilibrium.QuadratureOrder2D = 8     
    $ Equilibrium.QuadratureOrder1D = 5     
    $ Equilibrium.ext_maxiter = 5            
    $ Equilibrium.ext_tol = 1.0e-3           
    $ Equilibrium.int_maxiter = 10           
    $ Equilibrium.int_tol = 1.0e-4    

and initialise 

    $ Equilibrium.InitialiseParameters()

### **II. Computational domain mesh**

Folder **MESHES** contains several computational domain meshes on which the simulation can be run and from which the user may select one. 
A reasonable criterion for determining mesh size limits and general shape would be to select/define a mesh that encompasses the tokamak's first wall geometry while excluding the external magnetic coils.
Assign the desired mesh to the equilibrium's *MESH* object as follows:

    $ Equilibrium.MESH = Mesh('TRI06-SUPERFINE-REC') 

All proposed meshes have been generated using software GiD (https://www.gidsimulation.com/). 

### **III. Tokamak device**

After selecting an adequate computational domain mesh, the user must provide the tokamak's geometry data and use it to declare simulation object **Tokamak** (contained in src/Tokamak.py): 

#### First wall 

First, generate a mesh object such that its boundary corresponds to the tokamak's first wall, for instance:

    $ TokamakFirstWallMesh = Mesh('TRI03-FINE-ITFW')

#### Magnets 

Then, generate the different tokmak's external magnets using the different available classes (see file src/Magnet.py), for instance:

    $ coil1 = QuadrilateralCoil(name = 'PF1',
    $                           Itotal = 5.73e6,
    $                           Xcenter = np.array([3.9431,7.5741]),
    $                           Area = 0.25)
    $ coil2 = QuadrilateralCoil(name = 'PF2',
    $                           Itotal= -2.88e6,
    $                           Xcenter = np.array([8.2851,6.5398]),
    $                           Area = 0.25)

and place them inside a list

    $ magnets = [coil1, coil2]

#### Generate Tokamak

Finally, generate the **Tokamak** object and assign it to equilibrium object *TOKAMAK* as follows: 

    $ Equilibrium.TOKAMAK = Tokamak(WALL_MESH = TokamakFirstWallMesh, MAGNETS = magnets)

Defining a tokamak object is actually optional for the **FIXED-boundary** problem, however we recommend providing the tokamak's first wall mesh. 
On the other hand, defining external magnets is mandatory for the **FREE-boundary** problem.

### **IV. Initial plasma boundary**

The next step consists in defining the plasma cross-section initial boundary using class **InitialPlasmaBoundary** (contained in src/InitialPlasmaBoundary.py). 

While for the **FIXED-boundary** problem this initial boundary will not change throughout the simulation, for the **FREE-boundary** case such domain is simply an initial guess that will evolve and converge iteratively towards the equilibrium state. 
Different models for the initial plasma boundary level-set function are already implemented inside class **InitialPlasmaBoundary**, nonetheless the user is free to parametrise a new one. 

For instance, the following lines initiate the plasma boundary level-set function using a parametrised cubic hamiltonian model:

    $ X_SADDLE = np.array([5.2, -2.9])       # ACTIVE SADDLE POINT   
    $ X_RIGHT = np.array([7.9, 0.6])         # POINT ON RIGHT
    $ X_LEFT = np.array([4.5, 1.5])          # POINT ON LEFT
    $ X_TOP = np.array([5.9, 3.7])           # POINT ON TOP
    $ 
    $ Equilibrium.initialPHI = InitialPlasmaBoundary(EQUILIBRIUM = Equilibrium,
    $                                                GEOMETRY = 'cubicHam',
    $                                                Xsaddle = X_SADDLE,  # ACTIVE SADDLE POINT        
    $                                                Xright = X_RIGHT,    # POINT ON RIGHT
    $                                                Xleft = X_LEFT,      # POINT ON LEFT
    $                                                Xtop = X_TOP)        # POINT ON TOP

A special case arises in the FIXED-boundary problem when the initial plasma boundary coincides with the computational mesh boundary. In this scenario, the entire mesh corresponds to a fixed plasma domain with its boundary aligned to the mesh nodes, allowing the problem to be solved using a standard finite element method (FEM) scheme.

Once the initial plasma boundary has been defined, the computational domain is ready to be discretised with

    $ Equilibrium.DomainDiscretisation(INITIALISATION = True)

### **V. Initial plasma magnetic flux field (initial guess)**

Similarly to the initial plasma boundary, the user must then provide an initial guess for the poloidal magnetic flux by defining an object of class **InitialGuess** (contained in src/InitialPSIGuess.py).
Several parametrised models are already implemented. 
Similarly to the previous section, in the following example the initial guess is taken as parametrised cubic hamiltonian so that plasma boundary and normalised initial guess are in correspondance: 

    $ X_SADDLE = np.array([5.2, -2.9])        # ACTIVE SADDLE POINT
    $ X_RIGHT = np.array([7.9, 0.6])          # POINT ON RIGHT
    $ X_LEFT = np.array([4.5, 1.5])           # POINT ON LEFT
    $ X_TOP = np.array([5.9, 3.7])            # POINT ON TOP
    $
    $ X0 = list()
    $ X0.append(np.array([6.0,0.0],dtype=float))
    $ Equilibrium.initialPSI = InitialGuess(EQUILIBRIUM = Equilibrium,
    $                                       PSI_GUESS = 'cubicHam',
    $                                       NORMALISE = True,
    $                                       Xsaddle = X_SADDLE,  # ACTIVE SADDLE POINT        
    $                                       Xright = X_RIGHT,    # POINT ON RIGHT
    $                                       Xleft = X_LEFT,      # POINT ON LEFT
    $                                       Xtop = X_TOP,        # POINT ON TOP
    $                                       X0 = X0)     

With the initial already defined, the unknown arrays are initialised with 

    $ Equilibrium.InitialisePSI()

### **VI. Plasma toroidal current model**

Finally, the user must provide the model for the toroidal plasma current model appearing in the GS equation source term.  
This consists simply in defining an object of class **CurrentModel** (contained in src/PlasmaCurrent.py).
Several parametrised models are already implemented.

For instance, the following example defines the plasma current using the model used in the APEC plasma solver. 

    $ Equilibrium.PlasmaCurrent = CurrentModel(EQUILIBRIUM = Equilibrium,
    $                                          MODEL = 'APEC',
    $                                          Ii = 0.81,         # PLASMA INTERNAL INDUCTANCE
    $                                          Betap = 0.75,      # POLOIDAL BETA
    $                                          R0 = 6.0,          # MEAN RADIUS
    $                                          Tcurrent = 15e6)   # TOTAL PLASMA CURRENT

### **VII. Launch simulation**

Once all previous steps have been executed, the user may call the solver in order to solve the defined equilibrium problem in the following way:

    $ Equilibrium.EQUILI("TS-CASE")

This command will launch the python plasma equilibrium solver with the defined problem parameters. 
The user may find the simulation results inside a new folder *RESULTS*, that will be created in the same code directory if needed, inside a new folder with name "TS-CASE" + mesh name. 





