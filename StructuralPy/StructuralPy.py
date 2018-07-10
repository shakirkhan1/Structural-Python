import MKE as sa
import Graphics as gr
import numpy as np
from scipy.interpolate import interp1d


def ui():
    """Priant main program info."""


    print()
    print("========================================")
    print("Structural Analysis Program     Version  1.00")
    print("Copyright (C) 2018    Igor Banfi")
    print()
    print("This software comes with ABSOLUTELY NO WARRANTY,\n    subject to the GNU General Public License.")
    print()
    print()
    print()
    print("========================================")
    print()


def options1():
    """Prints first options menu."""
    while True:
        print("===============================================")
        print('{0:10}  {1:10}'.format("Load", "Load existing configuration"))
        print('{0:10}  {1:10}'.format("Oper", "Configure configuration"))
        print("===============================================")
        in_ = input(" >")
        try:
            if in_[0] == "o" or in_[0] == "O":
                oper()
            if in_[0] == "L" or in_[0] == "l":
                try:
                    filename = input("Input file name:\n >")
                    beams, supports, pointload, contload = load(filename)
                    oper(supports, beams, pointload, contload)
                except (IOError):
                    print("File not found")
        except (IndexError):
            pass

def load(filename):
    """Loads and parses selecte file."""

    file = open(filename, 'r')

    lines = file.readlines()

    lines_ = []
    for line in lines:
        lines_.append(line.split())

    indices = []
    for num, line in enumerate(lines_):
        if '#' in line:
            indices.append(num)

    lines = [x for x in lines_ if x != []]
    lines_ = [x for x in lines if '#' not in x[0]]

    beamsindex = lines_.index(["Beams"])
    supportsindex = lines_.index(["Supports"])
    pointloadsindex = lines_.index(["PointLoads"])
    contloadsindex = lines_.index(["ContLoads"])

    i = np.array([x for x in np.array([beamsindex, supportsindex, pointloadsindex, contloadsindex]) - beamsindex])
    i_ = [v if v > 0 else 0 if v < 0 else 0 for v in i]
    try:
        next_ = min([x for x in i_ if x != 0])
        beams = lines_[beamsindex + 1:next_ + beamsindex]
    except IndexError:
        beams = lines_[beamsindex + 1::]
    beams_ = []
    for i in range(len(beams)):
        beams_.append([])
        beams_[i] = [beams[i][0], beams[i][1], beams[i][2], np.float(beams[i][3]), np.float(beams[i][4]),
                     float(beams[i][5]), float(beams[i][6]), int(beams[i][7]), int(beams[i][8])]

    i = np.array([x for x in np.array([beamsindex, supportsindex, pointloadsindex, contloadsindex]) - supportsindex])
    i_ = [v if v > 0 else 0 if v < 0 else 0 for v in i]
    try:
        next_ = min([x for x in i_ if x != 0])
        supports = lines_[supportsindex + 1:next_ + supportsindex]
    except IndexError:
        supports = lines_[supportsindex + 1::]
    supports_ = []
    for i in range(len(supports)):
        supports_.append([])
        supports_[i] = [supports[i][0], supports[i][1], float(supports[i][2]), np.float(supports[i][3]),
                        np.float(supports[i][4]), np.float(supports[i][5]), np.float(supports[i][6]),
                        np.float(supports[i][7])]

    i = np.array([x for x in np.array([beamsindex, supportsindex, pointloadsindex, contloadsindex]) - pointloadsindex])

    i_ = [v if v > 0 else 0 if v < 0 else 0 for v in i]
    try:
        next_ = min([x for x in i_ if x != 0])
        pointload = lines_[pointloadsindex + 1:next_ + pointloadsindex]
    except IndexError:
        pointload = lines_[pointloadsindex + 1::]
    pointload_ = []
    for i in range(len(pointload)):
        pointload_.append([])
        pointload_[i] = [pointload[i][0], float(pointload[i][1]), float(pointload[i][2]), float(pointload[i][3])]

    i = np.array([x for x in np.array([beamsindex, supportsindex, pointloadsindex, contloadsindex]) - contloadsindex])
    i_ = [v if v > 0 else 0 if v < 0 else 0 for v in i]
    try:
        next_ = min([x for x in i_ if x != 0])
        contload = lines_[contloadsindex + 1:next_ + contloadsindex]
    except (IndexError, ValueError):
        contload = lines_[contloadsindex + 1::]
    contload_ = []
    for i in range(len(contload)):
        contload_.append([])
        contload_[i] = [contload[i][0], contload[i][1], contload[i][2]]

    return beams, supports, pointload, contload


def oper(supports_=0, beams_=0, pointloads_=0, contloads_=0):
    """
    Main operations menu. Options are:
    -define beam
    -define support
    -define poinoading
    -define contloading
    -assemble equation
    -Execute calculations
    -Save final resoults
    -Save matrices
    -Graphical representation
    """
    a = False
    x_ = False

    def info():
        """Prints options."""
        print("===============================================")
        print('{0:10}  {1:10}'.format("Sup", "Define support"))
        print('{0:10}  {1:10}'.format("Beam", "Define Beam"))
        print('{0:10}  {1:10}'.format("Pointload", "Define pointload"))
        print('{0:10}  {1:10}'.format("Contload", "Define continous load"))
        print('{0:10}  {1:10}'.format("Assemble", "Assemble main matrices"))
        print('{0:10}  {1:10}'.format("X", "Execute calculations"))
        print('{0:10}  {1:10}'.format("Graphics", "Graphical presentation"))
        if a:
            print('{0:10}  {1:10}'.format("MSave", "Save matrices and vectors in file"))
        if x_:
            print('{0:10}  {1:10}'.format("FSave", "Save final values in file"))
        print("===============================================")

    def list_elements(supports__, beams__, pointloads__, contloads__):
        """Lists all defined elements."""
        print("Supports")
        print("{:<10} {:>10} {:>10} {:>10} {:>10}".format("#", "Name", "Type", "X", "Y"))
        for num_, sup_ in enumerate(supports__):
            print('{:<10} {:>10} {:>10} {:>10} {:>10}'.format(num_, sup_.id, sup_.type, sup_.X, sup_.Z))
        print("")
        print("Beams")
        print("{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} "
              "{:>10} {:>10}  {:>10}  {:>10}".format("#", "Name", "Supports1", "Supports2", "I",
                                                     "E", "A", "rho", "L", "num_ele", "Truss"))
        for num_, beam_ in enumerate(beams__):
            print("{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} "
                  "{:>10} {:>10} {:>10.4f}  {:>10}  {:>10}".format(num_, beam_.id,
                                                                   str([j.id for j in beam_.supports_1]),
                                                                   str([j.id for j in beam_.supports_2]),
                                                                   beam_.I, beam_.E, beam_.A, beam_.ro, beam_.L,
                                                                   beam_.num_ele, beam_.Truss))
        print("")
        print("Point loads")
        print("{:<10} {:>10} {:>10} {:>10} {:>10}".format("#", "Support id", "Fx", "Fy", "M"))
        for num_, pointload_ in enumerate(pointloads__):
            print("{:<10} {:>10} {:>10} {:>10} {:>10}".format(num_, pointload_[0], pointload_[1], pointload_[2],
                                                              pointload_[3]))
        print("")
        print("Cont loads")
        print("{:<10} {:>10} {:>10} {:>10}".format("#", "Beam id", "n", "q"))
        for num_, contload_ in enumerate(contloads__):
            print("{:<10} {:>10} {:>10} {:>10}".format(num_, contload_[0], str(contload_[1]), str((contload_[2]))))
    info()
    supports = []
    beams = []
    pointloads = []
    contloads = []
    try:
        for sup in supports_:
            name = sup[0]
            sup_type = sup[1]
            X = sup[2]
            Y = sup[3]
            k1 = sup[4]
            k2 = sup[5]
            kt = sup[6]
            alpha = sup[7]
            supports.append(sa.Support(name, sup_type, float(X), float(Y), k1=float(k1), k2=float(k2), kt=float(kt),
                                       alpha=float(alpha)))
    except TypeError:
        pass
    try:
        for beam in beams_:
            name = beam[0]
            supports_1 = beam[1].split(',')
            supports_2 = beam[2].split(',')
            index_1 = []
            index_2 = []
            supports_id = []
            for i in supports:
                supports_id.append(i.id)
            for i in supports_1:
                index_1.append(supports_id.index(i))
            for i in supports_2:
                index_2.append(supports_id.index(i))
            beams.append(sa.Beam(name,
                                 [supports[i] for i in index_1],
                                 [supports[i] for i in index_2],
                                 float(beam[3]),
                                 float(beam[4]),
                                 float(beam[5]),
                                 float(beam[6]),
                                 int(beam[7]), bool(int(beam[8]))))
    except TypeError:
        pass
    try:
        for pointload in pointloads_:
            support_id = pointload[0]
            Fx = float(pointload[1])
            Fy = float(pointload[2])
            M = float(pointload[3])
            pointloads.append([support_id, Fx, Fy, M])
    except TypeError:
        pass
    try:
        for contload in contloads_:
            beam_id = contload[0]
            n = contload[1].split(',')
            q = contload[2].split(',')
            n = np.array(n, dtype=float)
            q = np.array(q, dtype=float)
            contloads.append([beam_id, n, q])
    except TypeError:
        pass

    list_elements(supports, beams, pointloads, contloads)

    while True:
        in_ = input(" >")
        in_ = in_+" "
        if in_[0] == "S" or in_[0] == "s":
            name = input("Support id: \n >")
            sup_type = input("Support type: \n >")
            X = float(input("X: \n >"))
            Y = float(input("Y: \n >"))
            try:
                k1 = float(input("k1: \n >"))
            except ValueError:
                k1 = 0
            try:
                k2 = float(input("k2: \n >"))
            except ValueError:
                k2 = 0
            try:
                kt = float(input("kt: \n >"))
            except ValueError:
                kt = 0
            try:
                alpha = float(input("alpha: \n >"))
            except ValueError:
                alpha = 0
            try:
                supports[int(in_[1])] = sa.Support(name, sup_type, X, Y, k1, k2, kt, alpha)
            except (ValueError, IndexError):
                supports.append(sa.Support(name, sup_type, X, Y, k1, k2, kt, alpha))

        if in_[0] == "B" or in_[0] == "b":
            name = input("Beam id: \n >")
            supports_1 = input("Supports 1: \n >").split()
            supports_2 = input("Supports 2: \n >").split()
            index_1 = []
            index_2 = []
            supports_id = []
            for i in supports:
                supports_id.append(i.id)
            for i in supports_1:
                index_1.append(supports_id.index(i))

            for i in supports_2:
                index_2.append(supports_id.index(i))
            I = float(input("I: \n >"))
            E = float(input("E: \n >"))
            A = float(input("A: \n >"))
            ro = float(input("rho: \n >"))
            num_ele = int(input("num_ele: \n >"))
            Truss = int(input("Truss: \n >"))
            try:
                beams[int(in_[1])] = sa.Beam(name,
                                             [supports[i] for i in index_1],
                                             [supports[i] for i in index_2],
                                             I,
                                             E,
                                             A,
                                             ro,
                                             num_ele,
                                             Truss)
            except (ValueError, IndexError):
                beams.append(sa.Beam(name,
                                     [supports[i] for i in index_1],
                                     [supports[i] for i in index_2],
                                     I,
                                     E,
                                     A,
                                     ro,
                                     num_ele,
                                     Truss))

        if in_[0] == "P" or in_[0] == "p":
            support_id = input("Support id: \n >")
            Fx = float(input("Fx: \n >"))
            Fy = float(input("Fy: \n >"))
            M = float(input("M: \n >"))
            try:
                pointloads[int(in_[1])] = [support_id, Fx, Fy, M]
            except (ValueError, IndexError):
                pointloads.append([support_id, Fx, Fy, M])

        if in_[0] == "C" or in_[0] == "c":
            beam_id = input("Beam id: \n >")
            try:
                n = input("n: \n >").split(",")
                n = np.array(n, dtype=float)
            except TypeError:
                print("Inpute type Error. Subsitutet with 0.")
                n = np.array([0])
            try:
                q = input("q: \n >").split(",")
                q = np.array(q, dtype=float)
            except ValueError:
                print("Inpute type Error. Subsitutet with 0.")
                n = np.array([0])
            try:
                contloads[int(in_[1])] = [beam_id, n, q]
            except (ValueError, IndexError):
                contloads.append([beam_id, n, q])

        if in_[0] == "A" or in_[0] == "a":
            for contload in contloads:
                names = [i.id for i in beams]
                index_ = names.index(contload[0])
                if len(contload[1]) < 2:
                    n = [contload[1], contload[1]]
                else:
                    x = np.linspace(0, beams[index_].L, len(contload[1]))
                    n = interp1d(x, contload[1])
                if len(contload[2]) < 2:
                    q = [contload[2], contload[2]]
                else:
                    x = np.linspace(0, beams[index_].L, len(contload[2]))
                    q = interp1d(x, contload[2])
                beams[index_].set_loading(n, q)

            for beam in beams:
                beam.transformation_matrix()
                beam.form_elements()
                beam.form_M()
                beam.form_K()

            for pointload in pointloads:

                names = [i.id for i in supports]
                index_ = names.index(pointload[0])
                supports[index_].set_boundryforces(0, 0, 0, pointload[1], pointload[2], pointload[3])
            M, b1, Naming, M_ = sa.form_mainMatrix(beams, supports, dynamic=True)
            a = True

        if in_[0] == "X" or in_[0] == "x":
            r = np.linalg.solve(M, b1)
            R, Naming_new = sa.resoults_parsing(beams, supports, r, Naming)
            for beam, R_ in zip(beams, R):
                beam.calculate_forces(R_)
            for beam in beams:
                print(beam.id)
                print("u")
                print(beam.u)
                print("w")
                print(beam.w)
                print("N")
                print(beam.N)
                print("T")
                print(beam.T)
                print("M")
                print(beam.M)
            print()
            print("===============================================")
            print()
            NF, eigenvectors = sa.natural_frequencies(M_, M)
            NF = np.sort(NF)
            print("Natural frequences:")
            for num in range(5):
                print("f{}: {}".format(num, NF[num]))
            x_ = True
        if x_ and (in_[0] == "F" or in_[0] == "f"):
            file = open("FinalResoult.txt", "w+")
            file.write("\n")
            file.write("===============================================\n")
            file.write("\n")
            file.write("Final resoult\n")
            file.write("\n")
            for num, beam in enumerate(beams):
                file.write("\n")
                file.write("Beam\n")
                file.write("\n")
                file.write("{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} "
                      "{:>10} {:>10}  {:>10}  {:>10}\n".format("#", "Name", "Supports1", "Supports2", "I",
                                                               "E", "A", "rho", "L", "num_ele", "Truss"))
                file.write("{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} "
                           "{:>10} {:>10} {:>10.4f}  {:>10}  {:>10}\n".format(num, beam.id,
                                                                       str([i.id for i in beam.supports_1]),
                                                                       str([i.id for i in beam.supports_2]),
                                                                       beam.I, beam.E, beam.A, beam.ro, beam.L,
                                                                       beam.num_ele, beam.Truss))
                file.write("\n")
                file.write("\n")
                file.write("{:<20}  {:<25}  {:<25}  {:<25}  {:<25}  {:<25}  {:<25}\n".format("x/L", "u", "w", "fi", "N", "T", "M"))
                file.write("\n")
                xs = np.linspace(0, 100, beam.num_ele, endpoint=True)
                for xs_, u, w, fi, N, T, M in zip(xs, beam.u, beam.w, beam.fi, beam.N, beam.T, beam.M):
                    file.write("%{:<19.2f}  {:<25}  {:<25}  {:<25}  {:<25}  {:<25}  {:<25}\n".format(xs_, u, w, fi, N, T, M))

            file.close()
        if a and (in_[0] == "M" or in_[0] == "m"):
            np.savetxt("StiffnessMatrix.mat", M, delimiter=',')
            np.savetxt("MassMatrix.mat", M_, delimiter=',')
            np.savetxt("Loading.vec", b1, delimiter=',')
            np.savetxt("Naming.vec", Naming, delimiter=',', fmt="%s")

        if in_[0] == "g" or in_[0] == "G":
            draw_process(beams, supports)

        else:
            info()
            list_elements(supports, beams, pointloads, contloads)


def draw_process(beams, supports):
    """Draw process."""

    def info():
        print("===============================================")
        print('{0:10}  {1:10}'.format("L", "Loading"))
        print('{0:10}  {1:10}'.format("u", "Axial deformation"))
        print('{0:10}  {1:10}'.format("w", "Deflection"))
        print('{0:10}  {1:10}'.format("N", "Tension"))
        print('{0:10}  {1:10}'.format("T", "Shear"))
        print('{0:10}  {1:10}'.format("M", "Bending Moment"))
        print('{0:10}  {1:10}'.format("TAB", "Legend"))
        print('{0:10}  {1:10}'.format("ESC", "Quit graphical"))
        print("===============================================")

    info()

    def draw1():
        global draw
        draw = gr.Drawing()
        draw.scaling(beams)
        for beam_ in beams:
            draw.draw_beam(beam_)
        for sup_ in supports:
            draw.draw_support(sup_)

    u_, w_, N_, T_, M_, q_, n_ = [], [], [], [], [], [], []

    for beam in beams:
        u_ = np.hstack((u_, beam.u))
        w_ = np.hstack((w_, beam.w))
        q_ = np.hstack((q_, np.array(beam.q_).flatten()))
        n_ = np.hstack((n_, np.array(beam.n_).flatten()))
        try:
            N_ = np.hstack((N_, beam.N))
            T_ = np.hstack((T_, beam.T))
            M_ = np.hstack((M_, beam.M))
        except:
            pass
    u_ = np.hstack((u_, 0))
    w_ = np.hstack((w_, 0))
    n_ = np.hstack((n_, 0))
    q_ = np.hstack((q_, 0))
    N_ = np.hstack((N_, 0))
    T_ = np.hstack((T_, 0))
    M_ = np.hstack((M_, 0))

    size_u = max(u_) - min(u_)
    size_w = max(w_) - min(w_)
    size_n = max(n_) - min(n_)
    size_q = max(q_) - min(q_)
    size_N = max(N_) - min(N_)
    size_T = max(T_) - min(T_)
    size_M = max(M_) - min(M_)

    F_ = []
    for sup in supports:
        F_.append(np.sqrt(sup.Fx**2 + sup.Fy**2))
        F_.append(0)

    size_F = max(F_) - min(F_)

    draw1()
    u, w, T, N, M, l, Legend = False, False, False, False, False, False, False
    running = True
    while running:
        for event in gr.pygame.event.get():
            if event.type == gr.pygame.QUIT or event.type == gr.pygame.KEYDOWN and event.key == gr.pygame.K_ESCAPE:
                running = False
                print("KONEC")
            if event.type == gr.pygame.KEYDOWN:
                if event.type == gr.pygame.KEYDOWN and event.key == gr.pygame.K_s:
                    gr.pygame.image.save(draw.screen, "Structural_analysis.jpg")
                if event.type == gr.pygame.KEYDOWN and event.key == gr.pygame.K_u:
                    u = ~u

                if event.type == gr.pygame.KEYDOWN and event.key == gr.pygame.K_l:
                    l = ~l

                if event.type == gr.pygame.KEYDOWN and event.key == gr.pygame.K_w:
                    w = ~w

                if event.type == gr.pygame.KEYDOWN and event.key == gr.pygame.K_t:
                    T = ~T

                if event.type == gr.pygame.KEYDOWN and event.key == gr.pygame.K_n:
                    N = ~N

                if event.type == gr.pygame.KEYDOWN and event.key == gr.pygame.K_m:
                    M = ~M
                if event.type == gr.pygame.KEYDOWN and event.key == gr.pygame.K_TAB:
                    Legend = ~Legend

                draw1()

                if l:
                    for beam in beams:
                        if size_q:
                            size = (max(np.hstack((beam.q_, 0))) - min(np.hstack((beam.q_, 0)))) / size_q * 100
                            draw.draw_loading(beam.q_, beam, 1, draw.RED, draw_size=size)
                        if size_n:
                            size = (max(beam.n_) - min(beam.n_)) / size_n * 100
                            draw.draw_loading(beam.n_, beam, -1, draw.BLACK, draw_size=size)

                    for sup in supports:
                        if size_F:
                            size = (sup.Fx**2 + sup.Fy**2)/size_F**2
                        else:
                            size = 0
                        draw.draw_boundryloading(sup, size*100, 20)

                if u:
                    for beam in beams:
                        size = (max(np.hstack((beam.u, 0))) - min(np.hstack((beam.u, 0))))/size_u * 100
                        draw.draw_loading(beam.u, beam, 1, draw.BLUE, draw_size = size)
                if w:
                    for beam in beams:
                        size = (max(np.hstack((beam.w, 0))) - min(np.hstack((beam.w, 0))))/size_w * 100
                        draw.draw_loading(beam.w, beam, -1, draw.GREEN, draw_size = size)
                if N:
                    for beam in beams:
                        size = (max(np.hstack((beam.N, 0))) - min(np.hstack((beam.N, 0))))/size_N * 100
                        draw.draw_loading(beam.N, beam, 1, draw.DARKOLIVEGREEN, draw_size = size)
                if T:
                    for beam in beams:
                        size = (max(np.hstack((beam.T, 0))) - min(np.hstack((beam.T, 0))))/size_T * 100
                        draw.draw_loading(beam.T, beam, -1, draw.LBLUE, draw_size = size)
                if M:
                    for beam in beams:
                        size = (max(np.hstack((beam.M, 0))) - min(np.hstack((beam.M, 0))))/size_M * 100
                        draw.draw_loading(beam.M, beam, 1, draw.PURPLE, draw_size = size)
                if Legend:
                    draw.legend()
            gr.pygame.display.update()


if __name__.endswith('__main__'):
    ui()
    options1()





