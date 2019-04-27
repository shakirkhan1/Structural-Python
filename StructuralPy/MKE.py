import numpy as np
from scipy.linalg import block_diag
import scipy.integrate as integrate


class Support(object):
    """A support which represents an endpoint of a beam/beams.
    """

    def __init__(self, id, type, X, Z, k1=0, k2=0, kt=0, alpha=0):
        """
        Attributes:
        id: A string which identifies a support.
        type: pinned, fixed, roller, joint, endpoint, beam_roller, slider_on_beam, joint_slider_on_beam
        k1: spring stifness direction x (default is 0)
        k2: spring stifness direction y (default is 0)
        kt: torsion string stifness (default is 0)
        alpha: angle of support [degrees] (default is 0)
        """
        self.id = id
        self.type = type
        self.X = X
        self.Z = Z
        self.k1 = k1
        self.k2 = k2
        self.kt = kt

        #Defining several variables with initial values
        self.u = 0
        self.w = 0
        self.Fx = 0
        self.Fy = 0
        self.M = 0
        self.fi = 0

        #Defining Node Freedom Signature
        self.set_NFS()

        #Defining transformation matrix of the support
        ##All supports in the same node must have the same orientation!!!
        alpha = float(alpha)/180*np.pi
        self.tran_m = np.array([[np.cos(alpha), np.sin(alpha)],
                       [-np.sin(alpha), np.cos(alpha)]])
    def __repr__(self):
        return "Item(%s)" % (self.id)
    def __eq__(self, other):
        if isinstance(other, Item):
            return (self.id == other.id)
        else:
            return False
    def __hash__(self):
        return hash(self.__repr__())

    def set_NFS(self):
        """
        This method defines Node Freedom Signature (NFD) based on boundry conditions and consistency conditions.
        First three bits of NFD represent freedom of movement in 3 DOF: x, y, fi
        If a value of individual bit is 0, the movement of node is limited in belonging DOF. (x = 0)
        If a value of individual bit is 1, the value of secondary variable for belonging DOF is 0. (Fx = 0)
        Last three bits represent consistency condition in 3 DOF: x, y, fi
        If a value of individual bit is 0, the movement of beams connected via the node is not consisten. (x1 != x2)
        If a value of individual bit is 1, the movement of beams connected via the node is consistent. (x1 = x2)
        """

        if self.type == "pinned":
            self.NFD = (0, 0, 1, 0, 0, 0)

        if self.type == "fixed":
            self.NFD = (0, 0, 0, 0, 0, 0)

        if self.type == "roller":
            self.NFD = (1, 0, 1, 1, 0, 0)

        if self.type == "joint":
            self.NFD = (1, 1, 1, 1, 1, 0)

        if self.type == "endpoint":
            self.NFD = (1, 1, 1, 1, 1, 1)

        if self.type == "beam_roller":
            self.NFD = (1, 0, 1, 1, 0, 1)

        if self.type == "slider_on_beam":
            self.NFD = (1, 1, 1, 0, 1, 1)

        if self.type == "joint_slider_on_beam":
            self.NFD = (1, 1, 1, 0, 1, 0)


    def set_boundryforces(self, u, w, fi, Fx, Fy, M):
        """Sets deflection, forces and/or moment acting on the support.
        Attributes:
        u: deflection in x DOF
        w: deflection in y DOF
        fi: deflection in fi DOF
        Fx: force applied in x direction
        Fy: force applied in y direction
        M: Moment applied
        """
        self.u = u
        self.w = w
        self.fi = fi
        self.Fx = Fx
        self.Fy = Fy
        self.M = M

class Beam(object):
    """A beam which represents bearing element supported by supports. Boundry conditions are enforced by supportes.
    Beam has 2 supports.
    """

    def __init__(self, id, supports_1, supports_2, I, E, A, ro = 1, num_ele = 3, Truss = False):
        """
        Attributes:
            id: A string which identifies a beam
            supports_1: Tuple which appoints supports at node 1
            supports_2: Tuple which appoints supports at node -1
            I: Second moment of area
            E: Elasticity module
            A: Cross section area
            ro: Density
            num_ele: number of finite elements
            Truss: True if beam is a truss. Truss has no bending moments.
        """
        self.id = id
        self.supports_1 = supports_1
        self.supports_2 = supports_2
        self.I = I
        self.E = E
        self.A = A
        self.ro = ro
        self.num_ele = int(num_ele)
        self.Truss = Truss

        #We define couple placeholding variables
        self.u = (0,)
        self.w = (0,)
        self.n, self.n_ = [0, 0], [0, 0]
        self.q, self.q_ = [0, 0], [0, 0]
        self.N = (0,)
        self.T = (0,)
        self.M = (0,)
        self.force_N = np.zeros(self.num_ele)
        self.force_T = np.zeros(self.num_ele)
        self.force_M = np.zeros(self.num_ele)


        #Length is calculated by considering support positions
        self.L = np.sqrt((supports_1[0].X - supports_2[0].X)**2 + (supports_1[0].Z - supports_2[0].Z)**2)

        #Lengh of individual finite element is calculated by considering beam lenght and number of elements
        self.h = float(self.L) / (float(self.num_ele))

        #Defining number of variables. (3 variables per node)
        self.Num_var = self.num_ele*3 + 3

    def transformation_matrix_to_global(self):
        """Forms and returns transformation matrix for transformation between local and global coordinates"""

        #Defining transformation matrix for end nodes
        self.tran_m_beam_global = np.identity(3)
        self.tran_m_beam_global[0:2, 0:2] = np.transpose(self.tran_m_beam)

        # Defining main transformation matrix for transformation from local to global
        self.tran_m_local_to_global = block_diag(*[self.tran_m_beam_global for i in range(self.num_ele + 1)])


    def transformation_matrix(self):
        """Forms and returns transformation matrix of whole beam."""

        #Distance between end nodes of beam in in global coordinates
        self.dX = self.supports_2[0].X - self.supports_1[0].X
        self.dZ = self.supports_2[0].Z - self.supports_1[0].Z

        #Defining transformation matrix for end nodes
        self.tran_m_beam = np.array([[self.dX / (np.sqrt(self.dX ** 2 + self.dZ ** 2)), self.dZ / (np.sqrt(self.dX ** 2 + self.dZ ** 2))],
                           [-self.dZ / (np.sqrt(self.dX ** 2 + self.dZ ** 2)), self.dX / (np.sqrt(self.dX ** 2 + self.dZ ** 2))]
                           ])

        #Considring orientation of supports
        tran_m_1 = np.matmul(self.supports_1[0].tran_m, self.tran_m_beam)
        tran_m_2 = np.matmul(self.supports_2[0].tran_m, self.tran_m_beam)

        #Defining main transformation matrix
        self.tran_m = np.identity(self.Num_var)
        self.tran_m[0:2, 0:2] = tran_m_1
        self.tran_m[-3:-1, -3:-1] = tran_m_2

    def psi0(self, x, L):
        """one of Galerkin functios used in integrating continuus load."""
        return 1 - x / L

    def psi1(self, x, L):
        """one of Galerkin functios used in integrating continuus load."""

        return 1 - 3 * x ** 2 / L ** 2 + 2 * x ** 3 / L ** 3

    def psi2(self, x, L):
        """one of Galerkin functios used in integrating continuus load."""

        return x - 2 * x ** 2 / L + x ** 3 / L ** 2

    def psi3(self, x, L):
        """one of Galerkin functios used in integrating continuus load."""

        return x / L

    def psi4(self, x, L):
        """one of Galerkin functios used in integrating continuus load."""

        return 3 * x ** 2 / L ** 2 - 2 * x ** 3 / L ** 3

    def psi5(self, x, L):
        """one of Galerkin functios used in integrating continuus load."""

        return -x ** 2 / L + x ** 3 / L ** 2

    def stiff_matrix(self, A, E, I, L):
        """Forms stiffnes matrix if an element."""

        K = np.array([[A * E / L,  0, 0, -A * E / L, 0, 0],
                      [0, 12 * E * I / L ** 3, 6 * E * I / L ** 2, 0,  -12 * E * I / L ** 3, 6 * E * I / L ** 2],
                      [0, 6 * E * I / L ** 2, 4 * E * I / L, 0, -6 * E * I / L ** 2, 2 * E * I / L],
                      [-A * E / L, 0, 0,  +A * E / L, 0, 0],
                      [0, -12 * E * I / L ** 3, -6 * E * I / L ** 2, 0,  12 * E * I / L ** 3, -6 * E * I / L ** 2],
                      [0, 6 * E * I / L ** 2, 2 * E * I / L, 0, -6 * E * I / L ** 2, 4 * E * I / L]])

        #If Truss shear and bending are removed
        if self.Truss == True:
            K[:, 1::3] = K[:, 1::3]*0
            K[:, 2::3] = K[:, 2::3]*0

        return K

    def set_loading(self, n, q):
        """Sets continuous loading along the beam."""
        self.n = n
        self.q = q

        #We form arrays of values for n and q along the beam. This is used for graphical representation.
        if callable(n):
            x = np.linspace(0, self.L, self.num_ele)
            self.n_ = n(x)
        elif not np.shape(n):
            self.n_ = [n, n]
        if callable(q):
            x = np.linspace(0, self.L, self.num_ele)
            self.q_ = q(x)
        if not np.shape(n):
            self.q_ = [q, q]
        else:
            self.q_ = q

    def mass_matrix(self):
        """Forms mass matrix of an element."""

        M = np.array([[140, 0, 0, 70, 0, 0],
                      [0, 156, 22*self.h, 0, 54, -13*self.h],
                      [0, 22*self.h, 4*self.h**2, 0, 13*self.h, -3*self.h**2],
                      [70, 0, 0, 140, 0, 0],
                      [0, 54, 13*self.h, 0, 156, -22*self.h],
                      [0, -13*self.h, -3*self.h**2, 0, -22*self.h, 4*self.h**2]])
        M = np.multiply(M, self.ro*self.A*self.h/420)

        return M

    def form_elements(self):
        """Forms stifness, mass matrixs and loading vectors for all elements"""

        #Stifness matrix for individual element
        self.K_ = self.stiff_matrix(self.A, self.E, self.I, self.h)
        #Mass matrix for individual element
        self.M_ = self.mass_matrix()

        #Forms loading vector for individual elements
        self.b_ = []
        for i in range(self.num_ele):
            xs = i*self.h
            b_ = self.form_b(self.n, self.q, xs)
            self.b_.append(b_)

    def form_K(self):
        """Forms main stifness matrix and main loaing vector for beam/truss."""

        #Combining loading vectors of all elements
        self.b = np.zeros(self.Num_var)
        for num, bi in enumerate(self.b_):
            self.b[num * 3:num * 3 + 6] += bi

        #Transforming loading vector in global coordinates
        self.b = np.matmul(np.matmul(np.transpose(self.tran_m), self.b), self.tran_m)

        #Combining stifness matrices of all elements
        self.Koc = np.zeros((self.Num_var, self.Num_var))
        for num in range(self.num_ele):
            self.Koc[num * 3:num * 3 + 6, num * 3:num * 3 + 6] += self.K_

        #Transforming stifness matrix in global coordinates
        self.K = np.matmul(np.matmul(np.transpose(self.tran_m), self.Koc), self.tran_m)

        #Forming Naming vector
        #[Beam id, main variable, node number, support id]
        Naming = []
        for i in range(self.num_ele + 1):
            Naming.append([self.id, 'u', i, '_'])
            Naming.append([self.id, 'w', i, '_'])
            Naming.append([self.id, 'fi', i, '_'])
        self.Naming_fe = np.array(Naming, dtype='U25')
        name = ""
        for sup in self.supports_1:
            name = name+sup.id+" "
        self.Naming_fe[0:3, 3] = name
        name = ""
        for sup in self.supports_2:
            name = name+sup.id+" "
        self.Naming_fe[-3:, 3] = name

    def form_M(self):
        """Forms main mass matrix for beam/truss"""

        #Forming mass matrix
        self.M = np.zeros((self.Num_var, self.Num_var))
        for num in range(self.num_ele):
            self.M[num * 3:num * 3 + 6, num * 3:num * 3 + 6] += self.M_

        #Transforming mass matrix in global coordinates
        self.M = np.matmul(np.matmul(np.transpose(self.tran_m), self.M), self.tran_m)

    def form_b(self, n, q, xs):
        """Forms loading vector for an element."""

        #Forming empty vector
        b = np.zeros(6)

        #Integrating axial continuous loading
        if callable(n):
            def n_(x_, L_, psi):
                return n(xs + x_) * psi(x_, L_)
        else:
            if type(n)==float:
                n = [n, n]
            def n_(x_, L_, psi):
                return (n[0] + (n[1] - n[0]) / L_ * x_) * psi(x_, L_)
        b[0] = integrate.quad(n_, 0, self.h, args=(self.h, self.psi0))[0]
        b[3] = integrate.quad(n_, 0, self.h, args=(self.h, self.psi3))[0]

        #integrating shear continuous loading
        if callable(q):
            def q_(x_, L_, psi):
                return q(xs + x_) * psi(x_, L_)
        else:
            if type(q)==float:
                q = [q, q]
            def q_(x_, L_, psi):
                return (q[0] + (q[1] - q[0]) / L_ * x_) * psi(x_, L_)
        b[1] = integrate.quad(q_, 0, self.h, args=(self.h, self.psi1))[0]
        b[2] = integrate.quad(q_, 0, self.h, args=(self.h, self.psi2))[0]
        b[4] = integrate.quad(q_, 0, self.h, args=(self.h, self.psi4))[0]
        b[5] = integrate.quad(q_, 0, self.h, args=(self.h, self.psi5))[0]

        return b

    def calculate_global_deflections(self, R):
        """Calculates deflection of beam in global coordinate system."""

        #Transforming back to beam coordinate system
        R1 = np.matmul(self.tran_m_local_to_global, np.matmul(self.tran_m, R))

        #Spliting according to variables
        R_ = np.reshape(R1, (self.num_ele + 1, 3))
        self.u_global = R_[:, 0]
        self.w_global = R_[:, 1]
        self.fi_global = R_[:, 2]

    def calculate_forces(self, R):
        """Calculates shear force, tension force and bending moment along the beam for each node."""

        #Transforming back to beam coordinate system
        R1 = np.matmul(self.tran_m, R)

        #Spliting according to variables
        R_ = np.reshape(R1, (self.num_ele + 1, 3))
        self.u = R_[:, 0]
        self.w = R_[:, 1]
        self.fi = R_[:, 2]

        #Spliting into individual elements
        R = []
        for i in range(len(R_) - 1):
            R.append(np.vstack((R_[i], R_[i + 1])))

        #Calculaintg forces and moment
        self.NTM = []
        for n, r in zip(self.b_, R):
            r = r.flatten()
            self.NTM.append(np.matmul(self.K_, r) - n)
        self.force_N = [-self.NTM[0][0]]
        self.force_T = [-self.NTM[0][1]]
        self.force_M = [-self.NTM[0][2]]
        for NTM in self.NTM:
            self.force_N.append(NTM[-3])
            self.force_T.append(NTM[-2])
            self.force_M.append(NTM[-1])

def form_mainMatrix(beams, supports, dynamic=False):
    """Equation assembler."""

    #Listing all matrices (stifness, laoding vectors, Naming vectors, mass matrices)
    K = block_diag(*[_.K for _ in beams])
    M = 0
    if dynamic:
        beam_massmatrices = [_.M for _ in beams]
        M = block_diag(*beam_massmatrices)
    b1 = np.hstack([_.b for _ in beams])
    Naming = np.vstack([_.Naming_fe for _ in beams])
    Naming = np.array(Naming, dtype='U25')

    def assembler_1(Sup, Naming, K, M, b1, var, dynamic=dynamic):
        """Function for finding and combining equations for the same node and variable."""
        indices = [num for num, i in enumerate(Naming) if var in i[1] and Sup.id in i[-1]]

        beams_name = ""
        for i_ in indices:
            beams_name = beams_name + " " + Naming[i_, 0]
        sups_inv = list(set([item for sublist in [Naming[i_, 3].split() for i_ in indices] for item in sublist]))
        sups_name = ""
        for sup_n in sups_inv:
            sups_name = sups_name + " " + sup_n
        New_name = [beams_name, var, '_', sups_name]
        for i in indices[1:]:
            K[:, indices[0]] += K[:, i]
        K = np.delete(K, indices[1:], axis=1)
        for i in indices[1:]:
            K[indices[0]] += K[i]
            b1[indices[0]] += b1[i]
        Naming[indices[0]] = New_name
        Naming = np.delete(Naming, indices[1:], axis=0)
        K = np.delete(K, indices[1:], axis=0)
        b1 = np.delete(b1, indices[1:], axis=0)
        if var=='u':
            K[indices[0], indices[0]] += -Sup.k1
        if var=='w':
            K[indices[0], indices[0]] += -Sup.k1
        if var=='fi':
            K[indices[0], indices[0]] += -Sup.kt
        if dynamic:
            for i in indices[1:]:
                M[:, indices[0]] += M[:, i]
            M = np.delete(M, indices[1:], axis=1)
            for i in indices[1:]:
                M[indices[0]] += M[i]
            M = np.delete(M, indices[1:], axis=0)
        return Naming, K, M, b1

    def assembler_2(Sup, Naming, K, M, b1, var, dynamic=dynamic):
        """Function for enforcing boundry condition."""
        indices_main_boundries = np.array(
            [num for num, i in enumerate(Naming) if var in i[1] and Sup.id in i[-1]]).flatten()
        Naming = np.delete(Naming, indices_main_boundries, axis=0)
        K = np.delete(K, indices_main_boundries, axis=0)
        K = np.delete(K, indices_main_boundries, axis=1)
        b1 = np.delete(b1, indices_main_boundries, axis=0)
        if dynamic:
            M = np.delete(M, indices_main_boundries, axis=0)
            M = np.delete(M, indices_main_boundries, axis=1)
        return Naming, K, M, b1

    for Sup in supports:
        if Sup.NFD[3] == 1:
            Naming, K, M, b1 = assembler_1(Sup, Naming, K, M, b1, "u")
        if Sup.NFD[4] == 1:
            Naming, K, M, b1 = assembler_1(Sup, Naming, K, M, b1, "w")
        if Sup.NFD[5] == 1:
            Naming, K, M, b1 = assembler_1(Sup, Naming, K, M, b1, "fi")
    for Sup in supports:
        if Sup.NFD[0] == 0:
            Naming, K, M, b1 = assembler_2(Sup, Naming, K, M, b1, 'u')
        if Sup.NFD[1] == 0:
            Naming, K, M, b1 = assembler_2(Sup, Naming, K, M, b1, 'w')
        if Sup.NFD[2] == 0:
            Naming, K, M, b1 = assembler_2(Sup, Naming, K, M, b1, 'fi')

    #Removing empty rows for calculation
    nonemptyrows = ~np.all(K == 0, axis=1)
    K = K[nonemptyrows]
    K = K[:, nonemptyrows]
    b1 = b1[nonemptyrows]
    Naming = Naming[nonemptyrows]
    M = M[nonemptyrows]
    M = M[:, nonemptyrows]

    #Set boundry conditions in loading vector
    for Sup in supports:
        indices = np.array([num for num, i in enumerate(Naming) if Sup.id in i[3]])
        for i_ in indices:
            if Naming[i_, 1] == 'u':
                b1[i_] += Sup.Fx
            if Naming[i_, 1] == 'w':
                b1[i_] += Sup.Fy
            if Naming[i_, 1] == 'fi':
                b1[i_] += Sup.M
    return K, b1, Naming, M


def resoults_parsing(beams, supports, r, Naming_r):
    """Splits calculated resoults according to beams/trusses"""

    naming = [_.Naming_fe for _ in beams]
    Naming = np.vstack(naming)
    Naming = np.array(Naming, dtype='U25')
    Resoult = np.zeros(len(Naming))
    for num_, Name_ in enumerate(Naming):
        for num, name in enumerate(Naming_r):
            if Name_[-1] == '_':
                if (Name_ == name).all():
                    Resoult[num_] = r[num]
            elif Name_[0] in name[0] and Name_[-1][0] in name[-1] and Name_[1] == name[1]:
                Resoult[num_] = r[num]
    Resoult_ = []
    i = 0
    for beam in beams:
        Resoult_.append(Resoult[i:i + beam.num_ele * 3 + 3])
        i += beam.num_ele * 3 + 3
    return Resoult_, Naming

def natural_frequencies(M, K):
    """Calculates natural frequencies of the construction."""
    A = np.matmul(np.linalg.inv(M), K)
    eigenvalues, eigvectors = np.linalg.eig(A)
    NF = np.sqrt(eigenvalues)
    return NF, eigvectors
