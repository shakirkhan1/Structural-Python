# Structural-Python
Strutrural Analysis  1.0
Igor Banfi


General Description
========================
Structural analysis of 2D structure by utilizing finite element method. Graphical presentation is achieved by PyGame.
Includes statical analysis under load and dynamic analysis. The major features are as follows:

Structural components
  Beams or Trusses
  Supports
  
Configuration description
  Keyword-driven geometry input file
  Beam or truss properties
    cross section properties, density
    number of finite elements

Structural calculations outputs
  Axial forces, Shear, Bending moments
  Axial deformation, Deflection
 
Dynamic output
  Natural undamped frequencies (eigenvalues)
  Vectors of individual natural frequence (eigenvectors)
  
Geometry File
========================
Structural python works with input files in plain text format. This file 
defines the geometry and properties of the system.

Coordinate system
------------------------
The file is based in SVG-like formate. 
|---------> X
|
|
|
V
Z


File format
------------------------
The input file is organized by keyword arguments:

Beams
-----------------------
Defines beam or truss element. Example:

Beams
#Name    Supports1    Supports2    I     E     A     rho     numEle    Truss
first     a,b          b,c         1     1     1      1         30       0

Name 
  (string)  
  defines the identification string of the Beam.

Supports1 
  (string or stirngs seperated by ',')  
  define the supports present in first node of the beam. Supports are identified by their id string.

Supports2 
  (string or stirngs seperated by ',') 
  define the supports present in second node of the beam. Supports are identified by their id string.

I
  (float)
  defines the second moment of inertia of the beam

E
  (float) 
  defines the Youngs module of the beam

A
  (float) 
  defines cross sectional area of the beam

rho
  (float)
  defines the density of the beam

numEle
  (int)
  defines number of finite elements used in the beam.

Truss
  (int) 
  defines if the element is beam or truss
  
  if Truss:
    #Truss
  else:
    #Beam


Supports
-----------------
Defines the support. Example

Supports
#Name    Type    X    Z     k1     k2     kt    alpha
a       pinned   0    0     0      0      0     180

Name 
  (string)
  defines the identification string of the Support.
  
Type
  (string)
  defines the type of support and its (NFS).
  
  Options:
    pinned 
    fixed 
    roller
    joint
    endpoint
    beam_roller
    slider_on_beam
    joint_slider_on_beam
    
X
  (float)
  defines X coordinate of the support
    
Z
  (float)
  defines Z coordinate of the support
  
k1
  (float)
  defines stifness of k1 spring
 
k2
  (float)
  defines stifness of k2 spring
  
kt
  (float)
  defines stifness of torsion spring

alpha
  (float, deg)
  deifnes angle of the suport.

Pointloads
-----------------
Defines the pointload on the support. Example

PointLoads
#SupportName  Fx   Fz   M
c             0    0    50000

SupportName
  (string)
  identifies the support by it id at wich the force/moment is applied

Fx
  (float)
  value of Fx force

Fz
  (float)
  value of Fz force

M
  (float)
  value of bending moment.
  
Contloads
-----------------
Defines the contload on the beam. Example

ContLoads
#BeamName  n q
first     100,100     2000,100

BeamName
  (string)
  identifies the beam on wich the load is applied.

n
  (float, list of floats seperated by ',')
  defines the value of axial distributed force loading.
  
q
  (float, list of floats seperated by ',')
  defines the value of shear distributed force loading.
  
Program Execution
========================

First menu:

===============================================
Load        Load existing configuration
Oper        Configure configuration
===============================================

Load:
  loads an input file.
  
Oper:
  starts without predefine dgeometry
  

Second menu

===============================================
Sup         Define support
Beam        Define Beam
Pointload   Define pointload
Contload    Define continous load
Assemble    Assemble main matrices
X           Execute calculations
Graphics    Graphical presentation
MSave       Save matrices and vectors in file
FSave       Save final values in file
===============================================

Sup:
  defines support

Beam:
  defines beam

Pointload:
  defines pointload

contload:
  defines contload

Assemble:
  assembles equations and matrices

X:
  calculates deformations and forces.
  
Graphics:
  Starts graphical presentation
 
MSave:
  saves matrices and vectors in file.
  
FSave:
  saves final calculations in file.


Graphics Menu

===============================================
L           Loading
u           Axial deformation
w           Deflection
N           Tension
T           Shear
M           Bending Moment
TAB         Legend
ESC         Quit graphical
===============================================

L:
  displays loading

u:
  displays axial deformation

w:
  displays deflection

N:
  displays tension force

T:
  displays shear force

M:
  displays bending moment

TAB:
  displays legend

ESC:
  exits graphical presentation
