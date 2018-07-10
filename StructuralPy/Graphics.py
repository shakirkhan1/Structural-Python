import pygame
import numpy as np
import pygame.gfxdraw

class Drawing():
    """Class for consturcing visual representation of construction."""

    def __init__(self):
        pygame.init()
        self.size = [800, 800]
        self.colours()
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption('Structural analysis')
        self.screen.fill(self.WHITE)
        pygame.display.flip()
        pygame.event.get()
        self.Scale = 100
        #self.font = pygame.font.SysFont('freesansbold.ttf', 20)
        self.font = pygame.font.SysFont("Sans", 15)

    def scaling(self, beams):
        """Used for determening size ratios and for scaling of the consturction."""
        Xmin = min(np.array([[beam.supports_1[0].X, beam.supports_2[0].X] for beam in beams]).flatten())
        Xmax = max(np.array([[beam.supports_1[0].X, beam.supports_2[0].X] for beam in beams]).flatten())

        Zmin = min(np.array([[beam.supports_1[0].Z, beam.supports_2[0].Z] for beam in beams]).flatten())
        Zmax = max(np.array([[beam.supports_1[0].Z, beam.supports_2[0].Z] for beam in beams]).flatten())
        self.Scale =  600/max((Zmax-Zmin, Xmax-Xmin))


    def colours(self):
        """Defines more commonly used colours."""

        self.BLACK = [  0,   0,   0]
        self.WHITE = [255, 255, 255]
        self.BLUE =  [  0,   0, 255]
        self.GREEN = [  0, 255,   0]
        self.RED =   [255,   0,   0]
        self.LBLUE = [0, 255, 255]
        self.PURPLE = [255, 0, 255]
        self.DARKOLIVEGREEN = [85, 107, 47]


    def text_objects(self, text):
        """
        Returns a surface with inputed text and rect.
        text: Text to output
        """
        textSurface = self.font.render(text, True, self.BLACK)
        return textSurface, textSurface.get_rect()

    def draw_beam(self, beam):
        """Draws a beam."""

        X1 = int(beam.supports_1[0].X*self.Scale)+100
        Z1 = int(beam.supports_1[0].Z*self.Scale)+100
        X2 = int(beam.supports_2[0].X*self.Scale)+100
        Z2 = int(beam.supports_2[0].Z*self.Scale)+100
        pygame.draw.line(self.screen, self.BLUE, [X1, Z1], [X2, Z2], 3)

        TextSurf, TextRect = self.text_objects(beam.id)
        TextRect.center = ((X1 + X2)/2 - 5*beam.tran_m_beam[0, 1], (Z2 + Z1)/2 - 5*beam.tran_m_beam[0, 0])
        TextSurf = pygame.transform.rotate(TextSurf, -np.arctan2(beam.dZ, beam.dX)*180/np.pi)
        self.screen.blit(TextSurf, TextRect)
        pygame.display.flip()
        pygame.event.get()

    def draw_support(self, support):
        """Draws a support based on type and orientation."""

        X = int(support.X*self.Scale)+100
        Z = int(support.Z*self.Scale)+100

        TextSurf, TextRect = self.text_objects(support.id)
        TextRect.center = (X-10*support.tran_m[0, 0] - 5*support.tran_m[0, 1],
                           Z-5*support.tran_m[1, 1] - 10*support.tran_m[1, 0])
        self.screen.blit(TextSurf, TextRect)



        if support.k1 != 0:
            pygame.draw.line(self.screen, self.BLACK,
                             [X, Z],
                             [X + 2 * support.tran_m[0, 0] - 5 * support.tran_m[0, 1],
                              Z + 2 * support.tran_m[1, 0] - 5 * support.tran_m[1, 1]], 2)
            pygame.draw.line(self.screen, self.BLACK,
                             [X + 2 * support.tran_m[0, 0] - 5 * support.tran_m[0, 1],
                              Z + 2 * support.tran_m[1, 0] - 5 * support.tran_m[1, 1]],
                             [X + 6 * support.tran_m[0, 0] + 5 * support.tran_m[0, 1],
                              Z + 6 * support.tran_m[1, 0] + 5 * support.tran_m[1, 1]], 2)
            pygame.draw.line(self.screen, self.BLACK,
                             [X + 6 * support.tran_m[0, 0] + 5 * support.tran_m[0, 1],
                              Z + 6 * support.tran_m[1, 0] + 5 * support.tran_m[1, 1]],
                             [X + 10 * support.tran_m[0, 0] - 5 * support.tran_m[0, 1],
                              Z - 5 * support.tran_m[1, 1] + 10 * support.tran_m[1, 0]], 2)

            pygame.draw.line(self.screen, self.BLACK,
                             [X + 10 * support.tran_m[0, 0] - 5 * support.tran_m[0, 1],
                              Z - 5 * support.tran_m[1, 1] + 10 * support.tran_m[1, 0]],
                             [X + 14 * support.tran_m[0, 0] + 5 * support.tran_m[0, 1],
                              Z + 14 * support.tran_m[1, 0] + 5 * support.tran_m[1, 1]], 2)
            pygame.draw.line(self.screen, self.BLACK,
                             [X + 14 * support.tran_m[0, 0] + 5 * support.tran_m[0, 1],
                              Z + 14 * support.tran_m[1, 0] + 5 * support.tran_m[1, 1]],
                             [X + 18 * support.tran_m[0, 0] - 5 * support.tran_m[0, 1],
                              Z + 18 * support.tran_m[1, 0] - 5 * support.tran_m[1, 1]], 2)
            pygame.draw.line(self.screen, self.BLACK,
                             [X + 18 * support.tran_m[0, 0] - 5 * support.tran_m[0, 1],
                              Z + 18 * support.tran_m[1, 0] - 5 * support.tran_m[1, 1]],
                             [X + 22 * support.tran_m[0, 0] + 5 * support.tran_m[0, 1],
                              Z + 22 * support.tran_m[1, 0] + 5 * support.tran_m[1, 1]], 2)
            pygame.draw.line(self.screen, self.BLACK,
                             [X + 22 * support.tran_m[0, 0] + 5 * support.tran_m[0, 1],
                              Z + 22 * support.tran_m[1, 0] + 5 * support.tran_m[1, 1]],
                             [X + 26 * support.tran_m[0, 0] - 5 * support.tran_m[0, 1],
                              Z + 26 * support.tran_m[1, 0] - 5 * support.tran_m[1, 1]], 2)

        if support.k2 != 0:
            pygame.draw.line(self.screen, self.BLACK,
                             [X, Z],
                             [X + 2 * support.tran_m[0, 1] - 5 * support.tran_m[0, 0],
                              Z + 2 * support.tran_m[1, 1] - 5 * support.tran_m[1, 0]], 2)
            pygame.draw.line(self.screen, self.BLACK,
                             [X + 2 * support.tran_m[0, 1] - 5 * support.tran_m[0, 0],
                              Z + 2 * support.tran_m[1, 1] - 5 * support.tran_m[1, 0]],
                             [X + 6 * support.tran_m[0, 1] + 5 * support.tran_m[0, 0],
                              Z + 6 * support.tran_m[1, 1] + 5 * support.tran_m[1, 0]], 2)
            pygame.draw.line(self.screen, self.BLACK,
                             [X + 6 * support.tran_m[0, 1] + 5 * support.tran_m[0, 0],
                              Z + 6 * support.tran_m[1, 1] + 5 * support.tran_m[1, 0]],
                             [X + 10 * support.tran_m[0, 1] - 5 * support.tran_m[0, 0],
                              Z - 5 * support.tran_m[1, 0] + 10 * support.tran_m[1, 1]], 2)

            pygame.draw.line(self.screen, self.BLACK,
                             [X + 10 * support.tran_m[0, 1] - 5 * support.tran_m[0, 0],
                              Z - 5 * support.tran_m[1, 0] + 10 * support.tran_m[1, 1]],
                             [X + 14 * support.tran_m[0, 1] + 5 * support.tran_m[0, 0],
                              Z + 14 * support.tran_m[1, 1] + 5 * support.tran_m[1, 0]], 2)
            pygame.draw.line(self.screen, self.BLACK,
                             [X + 14 * support.tran_m[0, 1] + 5 * support.tran_m[0, 0],
                              Z + 14 * support.tran_m[1, 1] + 5 * support.tran_m[1, 0]],
                             [X + 18 * support.tran_m[0, 1] - 5 * support.tran_m[0, 0],
                              Z + 18 * support.tran_m[1, 1] - 5 * support.tran_m[1, 0]], 2)
            pygame.draw.line(self.screen, self.BLACK,
                             [X + 18 * support.tran_m[0, 1] - 5 * support.tran_m[0, 0],
                              Z + 18 * support.tran_m[1, 1] - 5 * support.tran_m[1, 0]],
                             [X + 22 * support.tran_m[0, 1] + 5 * support.tran_m[0, 0],
                              Z + 22 * support.tran_m[1, 1] + 5 * support.tran_m[1, 0]], 2)
            pygame.draw.line(self.screen, self.BLACK,
                             [X + 22 * support.tran_m[0, 1] + 5 * support.tran_m[0, 0],
                              Z + 22 * support.tran_m[1, 1] + 5 * support.tran_m[1, 0]],
                             [X + 26 * support.tran_m[0, 1] - 5 * support.tran_m[0, 0],
                              Z + 26 * support.tran_m[1, 1] - 5 * support.tran_m[1, 0]], 2)

        if support.kt != 0:
            pygame.draw.arc(self.screen, self.BLACK, [X, Z - 15, 15, 15], np.pi / 2, np.pi, 2)
            pygame.draw.arc(self.screen, self.BLACK, [X - 10, Z - 15, 25, 15], 0, np.pi / 2, 2)
            pygame.draw.arc(self.screen, self.BLACK, [X - 10, Z - 30, 25, 40], 3 * np.pi / 2, 2 * np.pi, 2)
            pygame.draw.arc(self.screen, self.BLACK, [X - 28, Z - 28, 65, 40], np.pi, 3 * np.pi / 2, 2)

        if support.type == "pinned":
            pygame.draw.line(self.screen, self.BLACK,
                             [X - 10*support.tran_m[0, 0] + 20* support.tran_m[0, 1] ,
                              20*support.tran_m[1, 1] - 10 * support.tran_m[1, 0]  + Z],
                             [X, Z], 2)
            pygame.draw.line(self.screen, self.BLACK, [X, Z],
                             [X + 10* support.tran_m[0, 0] + 20 *support.tran_m[0, 1] ,
                              20*support.tran_m[1, 1] + 10 *  support.tran_m[1, 0] + Z], 2)
            pygame.draw.circle(self.screen, self.BLACK, [X, Z], 4)

            pygame.display.flip()
            pygame.event.get()


        elif support.type == "joint":
            pygame.draw.circle(self.screen, self.BLACK, [X, Z], 6)

            pygame.display.flip()
            pygame.event.get()


        elif support.type == "endpoint":
            pygame.draw.line(self.screen, self.BLACK, [X - 7, Z], [X + 7, Z], 2)
            pygame.draw.line(self.screen, self.BLACK, [X, Z - 7], [X, Z + 7], 2)
            pygame.draw.line(self.screen, self.BLACK, [X - 5, Z - 5], [X + 5, Z + 5], 2)
            pygame.draw.line(self.screen, self.BLACK, [X - 5, Z + 5], [X + 5, Z - 5], 2)

            pygame.display.flip()
            pygame.event.get()

        elif support.type == 'slider_on_beam':
            pygame.draw.line(self.screen, self.BLACK, [X - 8 * support.tran_m[0, 0] + 3 * support.tran_m[0, 1],
                                                       Z + 3 *support.tran_m[1, 1] -8 * support.tran_m[1, 0]],
                             [X + 8 * support.tran_m[0, 0] + 3 * support.tran_m[0, 1],
                                                       Z + 3 *support.tran_m[1, 1] +8 * support.tran_m[1, 0]], 2)

            pygame.draw.line(self.screen, self.BLACK, [X - 8 * support.tran_m[0, 0] - 3 * support.tran_m[0, 1],
                                                       Z - 3 *support.tran_m[1, 1] -8 * support.tran_m[1, 0]],
                             [X + 8 * support.tran_m[0, 0] - 3 * support.tran_m[0, 1],
                                                       Z - 3 *support.tran_m[1, 1] +8 * support.tran_m[1, 0]], 2)

        elif support.type == 'joint_slider_on_beam':
            pygame.draw.line(self.screen, self.BLACK, [X - 8 * support.tran_m[0, 0] + 3 * support.tran_m[0, 1],
                                                       Z + 3 *support.tran_m[1, 1] -8 * support.tran_m[1, 0]],
                             [X + 8 * support.tran_m[0, 0] + 3 * support.tran_m[0, 1],
                                                       Z + 3 *support.tran_m[1, 1] +8 * support.tran_m[1, 0]], 2)

            pygame.draw.line(self.screen, self.BLACK, [X - 8 * support.tran_m[0, 0] - 3 * support.tran_m[0, 1],
                                                       Z - 3 *support.tran_m[1, 1] -8 * support.tran_m[1, 0]],
                             [X + 8 * support.tran_m[0, 0] - 3 * support.tran_m[0, 1],
                                                       Z - 3 *support.tran_m[1, 1] +8 * support.tran_m[1, 0]], 2)
            pygame.draw.circle(self.screen, self.BLACK, [X, Z], 4)

        elif support.type == "roller":

            pygame.draw.line(self.screen, self.BLACK, [X - 10 * support.tran_m[0, 0] + 20 * support.tran_m[0, 1],
                                                       20 * support.tran_m[1, 1] - 10 * support.tran_m[1, 0] + Z],
                             [X, Z], 2)
            pygame.draw.line(self.screen, self.BLACK, [X, Z],
                             [X + 10 * support.tran_m[0, 0] + 20 * support.tran_m[0, 1],
                              20 * support.tran_m[1, 1] + 10 * support.tran_m[1, 0] + Z], 2)
            pygame.draw.ellipse(self.screen, self.BLACK,
                                [X - 7* support.tran_m[0, 0] + 8* support.tran_m[0, 1]
                                 , Z + 8 * support.tran_m[1, 1] + 7*support.tran_m[1, 0], 16, 16], 2)
            pygame.draw.circle(self.screen, self.BLACK, [X, Z], 4)
            pygame.display.flip()
            pygame.event.get()

        elif support.type == "beam_roller":

            pygame.draw.line(self.screen, self.BLACK, [X - 10 * support.tran_m[0, 0] + 20 * support.tran_m[0, 1],
                                                       20 * support.tran_m[1, 1] - 10 * support.tran_m[1, 0] + Z],
                             [X, Z], 2)
            pygame.draw.line(self.screen, self.BLACK, [X, Z],
                             [X + 10 * support.tran_m[0, 0] + 20 * support.tran_m[0, 1],
                              20 * support.tran_m[1, 1] + 10 * support.tran_m[1, 0] + Z], 2)
            pygame.draw.ellipse(self.screen, self.BLACK,
                                [X - 7* support.tran_m[0, 0] + 8* support.tran_m[0, 1]
                                , Z + 8 * support.tran_m[1, 1] - 7*support.tran_m[1, 0], 16, 16], 2)
            pygame.display.flip()
            pygame.event.get()


        elif support.type == "fixed":
            pygame.draw.rect(self.screen, self.BLACK, [X-15, Z-15, 30, 30], 3)

            pygame.display.flip()
            pygame.event.get()

    def draw_loading(self, loading, beam, direction, colour, draw_size=100):
        """Draws loading on the beam"""

        if (max(np.hstack((loading, 0)) - min(np.hstack((loading, 0))))):
            L = len(loading)
            loading = np.hstack((0, loading))
            loading = np.hstack((loading, 0))

            draw_param = draw_size / (max(loading) - min(loading)) * loading
            #draw_param = draw_param - min(draw_param)

            X1 = int(beam.supports_1[0].X * self.Scale) + 100
            Z1 = int(beam.supports_1[0].Z * self.Scale) + 100
            X2 = int(beam.supports_2[0].X * self.Scale) + 100
            Z2 = int(beam.supports_2[0].Z * self.Scale) + 100

            Parp_vec = np.array([(-Z2 + Z1)*direction, (X2 - X1)*direction]) / (np.sqrt((Z2 - Z1) ** 2 + (X2 - X1) ** 2))

            draw_param = np.matmul(draw_param.reshape(-1, 1), Parp_vec.reshape(-1, 2))
            x = np.linspace(X1, X2, L, endpoint=True).reshape(-1, 1)
            y = np.linspace(Z1, Z2, L, endpoint=True).reshape(-1, 1)
            Position = np.hstack((x, y))
            Position = np.vstack(([X1, Z1], Position, [X2, Z2]))
            draw_param = draw_param + Position
            pygame.draw.polygon(self.screen, colour, draw_param, 2)
            colour_ = np.array([i for i in colour])
            colour_ = np.hstack((colour_, 100))
            pygame.gfxdraw.filled_polygon(self.screen,draw_param, colour_)
            pygame.display.flip()
            pygame.event.get()


    def legend(self):
        """Draws legend of elements."""

        class TempSup():
            def __init__(self, id, type, X, Z, Fx, Fy, M, k1=0, k2=0, kt=0):
                self.id = id
                self.type = type
                self.X = X
                self.Z = Z
                self.tran_m = np.array([[1, 0],
                               [0, 1]])
                self.Fx = Fx
                self.Fy = Fy
                self.M = M
                self.k1 = k1
                self.k2 = k2
                self.kt = kt

        colour_ = np.hstack((np.array([i for i in self.WHITE]), 240))
        pygame.gfxdraw.filled_polygon(self.screen, [[0, 0], [self.size[0], 0], [self.size[0], self.size[1]], [0, self.size[1]]], colour_)

        # fixed support
        self.draw_support(TempSup("a", "fixed", 0/self.Scale, 0/self.Scale, 0, 0, 0))
        TextSurf, TextRect = self.text_objects("FIXED SUPPORT")
        TextRect.center = (200, 100)
        self.screen.blit(TextSurf, TextRect)

        # pinned support
        self.draw_support(TempSup("a", "pinned", 0/self.Scale, 100/self.Scale, 0, 0, 0))
        TextSurf, TextRect = self.text_objects("PINNED SUPPORT")
        TextRect.center = (200, 200)
        self.screen.blit(TextSurf, TextRect)

        # roller support
        self.draw_support(TempSup("a", "roller", 0/self.Scale, 200/self.Scale, 0, 0, 0))
        TextSurf, TextRect = self.text_objects("ROLLER SUPPORT")
        TextRect.center = (200, 300)
        self.screen.blit(TextSurf, TextRect)

        # joint support
        self.draw_support(TempSup("a", "joint", 0/self.Scale, 300/self.Scale, 0, 0, 0))
        TextSurf, TextRect = self.text_objects("JOINT SUPPORT")
        TextRect.center = (200, 400)
        self.screen.blit(TextSurf, TextRect)

        # endpoint support
        self.draw_support(TempSup("a", "endpoint", 0/self.Scale, 400/self.Scale, 0, 0, 0))
        TextSurf, TextRect = self.text_objects("ENDPOINT SUPPORT")
        TextRect.center = (200, 500)
        self.screen.blit(TextSurf, TextRect)

        # beam roller
        self.draw_support(TempSup("a", "beam_roller", 0/self.Scale, 500/self.Scale, 0, 0, 0))
        TextSurf, TextRect = self.text_objects("BEAM ROLLER")
        TextRect.center = (200, 600)
        self.screen.blit(TextSurf, TextRect)

        # slider on beam
        self.draw_support(TempSup("a", "slider_on_beam", 0/self.Scale, 600/self.Scale, 0, 0, 0))
        TextSurf, TextRect = self.text_objects("SLIDER ON BEAM")
        TextRect.center = (200, 700)
        self.screen.blit(TextSurf, TextRect)

        # joint slider on beam
        self.draw_support(TempSup("a", "joint_slider_on_beam", 200/self.Scale, 0/self.Scale, 0, 0, 0))
        TextSurf, TextRect = self.text_objects("JOINT SLIDER")
        TextRect.center = (400, 100)
        self.screen.blit(TextSurf, TextRect)

        # force loading
        self.draw_boundryloading(TempSup("a", "fixed", 200/self.Scale, 100/self.Scale, 1, 1, 0), 50, 20)
        TextSurf, TextRect = self.text_objects("FORCE LOADING")
        TextRect.center = (400, 200)
        self.screen.blit(TextSurf, TextRect)

        # moment loading
        self.draw_boundryloading(TempSup("a", "fixed", 200 / self.Scale, 200 / self.Scale, 0, 0, 1), 50, 20)
        TextSurf, TextRect = self.text_objects("MOMENT LOADING")
        TextRect.center = (400, 300)
        self.screen.blit(TextSurf, TextRect)

        # tension spring
        self.draw_support(TempSup(" ", "joint", 200/self.Scale, 300/self.Scale, 0, 0, 0, 1, 0, 0))
        TextSurf, TextRect = self.text_objects("TENSION SPRING")
        TextRect.center = (400, 400)
        self.screen.blit(TextSurf, TextRect)

        # torsion spring
        self.draw_support(TempSup(" ", "joint", 200/self.Scale, 400/self.Scale, 0, 0, 0, 0, 0, 1))
        TextSurf, TextRect = self.text_objects("TORSION SPRING")
        TextRect.center = (400, 500)
        self.screen.blit(TextSurf, TextRect)

        #axial continuous loading - n
        rect = pygame.Rect(0, 0, 20, 20)
        rect.center = (300, 600)
        colour_ = np.hstack((np.array([i for i in self.RED]), 100))
        pygame.draw.rect(self.screen, self.RED, rect, 2)
        pygame.gfxdraw.box(self.screen, rect, colour_)
        TextSurf, TextRect = self.text_objects("n")
        TextRect.center = (400, 600)
        self.screen.blit(TextSurf, TextRect)

        #shear continous loading - q
        rect = pygame.Rect(0, 0, 20, 20)
        rect.center = (300, 700)
        colour_ = np.hstack((np.array([i for i in self.BLACK]), 100))
        pygame.draw.rect(self.screen, self.BLACK, rect, 2)
        pygame.gfxdraw.box(self.screen, rect, colour_)
        TextSurf, TextRect = self.text_objects("q")
        TextRect.center = (400, 700)
        self.screen.blit(TextSurf, TextRect)

        #axial deformation - u
        rect = pygame.Rect(0, 0, 20, 20)
        rect.center = (500, 100)
        colour_ = np.hstack((np.array([i for i in self.BLUE]), 100))
        pygame.draw.rect(self.screen, self.BLUE, rect, 2)
        pygame.gfxdraw.box(self.screen, rect, colour_)
        TextSurf, TextRect = self.text_objects("u")
        TextRect.center = (600, 100)
        self.screen.blit(TextSurf, TextRect)

        # axial deformation - w
        rect = pygame.Rect(0, 0, 20, 20)
        rect.center = (500, 200)
        colour_ = np.hstack((np.array([i for i in self.GREEN]), 100))
        pygame.draw.rect(self.screen, self.GREEN, rect, 2)
        pygame.gfxdraw.box(self.screen, rect, colour_)
        TextSurf, TextRect = self.text_objects("w")
        TextRect.center = (600, 200)
        self.screen.blit(TextSurf, TextRect)

        # axial deformation - N
        rect = pygame.Rect(0, 0, 20, 20)
        rect.center = (500, 300)
        colour_ = np.hstack((np.array([i for i in self.DARKOLIVEGREEN]), 100))
        pygame.draw.rect(self.screen, self.DARKOLIVEGREEN, rect, 2)
        pygame.gfxdraw.box(self.screen, rect, colour_)
        TextSurf, TextRect = self.text_objects("N")
        TextRect.center = (600, 300)
        self.screen.blit(TextSurf, TextRect)

        # axial deformation - T
        rect = pygame.Rect(0, 0, 20, 20)
        rect.center = (500, 400)
        colour_ = np.hstack((np.array([i for i in self.LBLUE]), 100))
        pygame.draw.rect(self.screen, self.LBLUE, rect, 2)
        pygame.gfxdraw.box(self.screen, rect, colour_)
        TextSurf, TextRect = self.text_objects("T")
        TextRect.center = (600, 400)
        self.screen.blit(TextSurf, TextRect)

        # axial deformation - M
        rect = pygame.Rect(0, 0, 20, 20)
        rect.center = (500, 500)
        colour_ = np.hstack((np.array([i for i in self.PURPLE]), 100))
        pygame.draw.rect(self.screen, self.PURPLE, rect, 2)
        pygame.gfxdraw.box(self.screen, rect, colour_)
        TextSurf, TextRect = self.text_objects("M")
        TextRect.center = (600, 500)
        self.screen.blit(TextSurf, TextRect)

        pygame.display.flip()
        pygame.event.get()

    def draw_boundryloading(self, support, L, R):
        """Draws loading on the support."""

        X = int(support.X * self.Scale) + 100
        Z = int(support.Z * self.Scale) + 100

        if support.Fx != 0 or support.Fy != 0:
            M = L / np.sqrt(support.Fx ** 2 + support.Fy ** 2)
            X_ = X + int(support.Fx * M)
            Z_ = Z + int(support.Fy * M)


            X2 = (int((X - int(support.Fy * M) * 1.0)) + 9 * X_) / 10
            Z2 = int((Z + int(support.Fx * M)) * 1.0 + 9 * Z_) / 10
            X3 = int((X + int(support.Fy * M) * 1.0) + 9 * X_) / 10
            Z3 = int((Z - int(support.Fx * M)) * 1.0 + 9 * Z_) / 10

            pygame.draw.line(self.screen, self.RED, [X, Z], [X_, Z_], 3)
            pygame.draw.line(self.screen, self.RED, [X2, Z2], [X_, Z_], 3)
            pygame.draw.line(self.screen, self.RED, [X3, Z3], [X_, Z_], 3)

            pygame.display.flip()
            pygame.event.get()

        if support.M != 0:
            pygame.draw.arc(self.screen, self.RED, [X-R, Z-R, 2*R, 2*R], np.pi/4*3,
            np.pi / 4, 2)
            pygame.draw.line(self.screen, self.RED, [X+np.sqrt(2)/2*R, Z-np.sqrt(2)/2*R], [X+np.sqrt(2)/2*R+10, Z-np.sqrt(2)/2*R], 3)
            pygame.draw.line(self.screen, self.RED, [X+np.sqrt(2)/2*R, Z-np.sqrt(2)/2*R], [X+np.sqrt(2)/2*R, Z-np.sqrt(2)/2*R+10], 3)

            pygame.display.flip()
            pygame.event.get()