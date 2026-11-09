import pygame as pg
import numpy as np
import moderngl as mgl
import os
import warnings

from rendering.quaternion import Quaternion

dir_name = os.path.abspath(os.path.dirname(__file__))

def load_shader(ctx : mgl.Context, vertex_filename, frag_filename):
    with open(os.path.join(dir_name, "objects", vertex_filename)) as f:
        vertex_src = f.read()
    with open(os.path.join(dir_name, "objects", frag_filename)) as f:
        frag_src = f.read()
    
    return ctx.program(vertex_shader=vertex_src, fragment_shader=frag_src)

def load_mesh(ctx : mgl.Context, program, filename):
    v = []
    vn = []
    vertices = []
    indices = []

    with open(os.path.join(dir_name, "objects",filename), 'r') as file:
        for line in file:
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                vertex = [float(val) for val in values[1:4]]
                v.append(vertex)
            elif values[0] == 'vn':
                norm = [float(val) for val in values[1:4]]
                vn.append(norm)
            elif values[0] == 'vt':
                continue
            elif values[0] in ('usemtl', 'usemat'):
                continue
            elif values[0] == 'mtllib':
                continue
            elif values[0] == 'f':
                for val in values[1:]:
                    w = val.split('/')
                    vertex = np.hstack((v[int(w[0])-1], vn[int(w[2])-1]))
                    vertices.append(vertex)
                start = len(vertices) - len(values) + 1
                for i in range(start, len(vertices)-2):
                    indices.append([start, i+1, i+2])

    vbo = ctx.buffer(np.hstack(vertices).astype(np.float32).tobytes())
    ebo = ctx.buffer(np.hstack(indices).flatten().astype(np.uint32).tobytes())
    return ctx.simple_vertex_array(program, vbo, 'aPos', 'aNormal', index_buffer=ebo)

def perspective(fov, aspect, near, far):
    """
    -fov: field-of-view, larger values to see more of the scene
    -aspect: ratio (streching)
    -near/far: scaling
    """
    fov *= np.pi/180
    right = -np.tan(fov/2) * near
    top = -right / aspect
    return np.array([[near/right,0,0,0],
                     [0,near/top,0,0],
                     [0,0,(far+near)/(far-near),-2*far*near/(far-near)],
                     [0,0,1,0]], dtype=np.float32)

class Transform:
    def __init__(self):
        self._position = np.zeros(3)
        self._rotation = Quaternion()
        self.scale = 1
    
    @property
    def position(self):
        return self._position.copy()
    
    @position.setter
    def position(self, position):
        self._position[:] = position
    
    @property
    def x(self):
        return self._position[0]
    
    @x.setter
    def x(self, x):
        self._position[0] = x
    
    @property
    def y(self):
        return self._position[1]
    
    @y.setter
    def y(self, y):
        self._position[1] = y
    
    @property
    def z(self):
        return self._position[2]
    
    @z.setter
    def z(self, z):
        self._position[2] = z
    
    @property
    def rotation(self):
        return self._rotation.copy()
    
    @rotation.setter
    def rotation(self, rotation):
        self._rotation._arr[:] = rotation._arr
    
    @property
    def matrix(self):
        matrix = np.eye(4)
        matrix[:3,:3] = self._rotation.mat().dot(np.eye(3)*self.scale)
        matrix[:3,3] = self._position
        return matrix
    
    @property
    def inv_matrix(self):
        matrix = np.eye(4)
        matrix[:3,3] = -self._position
        scale = np.eye(4)
        scale[:3,:3] /= self.scale
        matrix = scale.dot(matrix)
        rot = np.eye(4)
        rot[:3,:3] = self.rotation.inv().mat()
        matrix = rot.dot(matrix)
        return matrix

    
class RenderObject:
    def __init__(self, vao):
        self.vao = vao

        # self.color = 1.0, 1.0, 1.0
        self.color = 1.0, 1.0, 1.0, 1.0

        self.transform = Transform()

        self.draw_mode = mgl.TRIANGLES
    
    def render(self):
        self.vao.program['model'] = tuple(np.hstack(self.transform.matrix.T))
        self.vao.program['color'] = self.color
        self.vao.render(self.draw_mode)

class Grid(RenderObject):
    def __init__(self, ctx : mgl.Context, program, n, spacing, centering=False):
        """
        -n:  number of squares
        -spacing: width of squares

        - low / high: shift placement of Grid
        """
        super().__init__(None)
        if centering:
            low = -(n-1)*spacing/2
            high = -low
        else:
            low = 0
            high = n-1
        vertices = []
        indices = []
        for i in range(n):
            if centering:
                vertices.append([low + spacing*i, 0, low])
                vertices.append([low + spacing*i, 0,  high])
            else:
                vertices.append([-spacing*i, 0, low])
                vertices.append([-spacing*i, 0,  high])
            indices.append([i*2, i*2+1])
        for i in range(n):
            if centering:
                vertices.append([low, 0, low + spacing*i])
                vertices.append([high, 0, low + spacing*i])
            else:
                vertices.append([low, 0, spacing*i])
                vertices.append([-high, 0, spacing*i])
            indices.append([n*2+i*2, n*2+i*2+1])
        vertices = np.hstack(vertices)
        indices = np.hstack(indices)
        vbo = ctx.buffer(vertices.astype(np.float32).tobytes())
        ebo = ctx.buffer(indices.astype(np.uint32).tobytes())
        self.vao = ctx.simple_vertex_array(program, vbo, 'aPos', index_buffer=ebo)
        self.draw_mode = mgl.LINES

class Text(RenderObject):
    TXT_COLORS = {"white": (255,255,255,255), "black": (0, 0,0,255), "transparent":(128,128,128,255), "blue": (0, 0,255,255), "red": (255, 0,0,255), "green": (0,255,0,255), "yellow": (255,255,0,255)}
    def __init__(self, ctx : mgl.Context, text:list=None, background:bool=False):
        super().__init__(None)
        self.text = text
        self.font = pg.font.SysFont('arial', 64)
        self.prog = ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_vert;
            in vec2 in_texcoord;
            out vec2 texcoord;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
                texcoord = in_texcoord;
            }
            """,
            fragment_shader="""
            #version 330
            uniform sampler2D texture;
            in vec2 texcoord;
            out vec4 fragColor;
            void main() {
                fragColor = texture2D(texture, texcoord);
            }
            """
        )
        self.set_objects(ctx, background)

        # vbo = ctx.buffer(vertices.tobytes())
        # self.vao = ctx.simple_vertex_array(prog, vbo, 'in_vert', 'in_texcoord')

        # text_surface = font.render(text, True, (255,0,0,255), (0, 0,0,255) if background else (128,128,128,255))

        # texture = ctx.texture((text_surface.get_width(), text_surface.get_height()), 4, pg.image.tostring(text_surface, "RGBA", True))
        # texture.use()

        self.draw_mode = mgl.TRIANGLE_STRIP

    def render(self):
        for i, v in enumerate(self.vaos):
            self.textures[i].use()
            v.render(self.draw_mode)

    def set_objects(self, ctx, background):
        """
        Properly set location of each text box.
        Columns 1,2 specify location of text box, range (-1,1)
        Columns 3,4 specify location of text inside box, range (0,1)
        """
        self.vertices = []
        self.vaos = []
        self.textures = []
        for tidx, txt in enumerate(self.text):
            if tidx == 0: #combat
                self.vertices.append(np.array([
                    -1.0, 0.92, 0.0, 0.0,
                    -1.0, 1.0, 0.0, 1.0,
                    -0.73, 0.92, 1.0, 0.0,
                    -0.73, 1.0, 1.0, 1.0
                ], dtype=np.float32))
                text_surface = self.font.render(txt, True, Text.TXT_COLORS["white"], Text.TXT_COLORS["black"] if background else Text.TXT_COLORS["transparent"])
                self.textures.append(ctx.texture((text_surface.get_width(), text_surface.get_height()), 4, pg.image.tostring(text_surface, "RGBA", True)))
            # elif tidx == 1: #teams
            #     self.vertices.append(np.array([
            #         -1.0, 0.82, 0.0, 0.0,
            #         -1.0, 0.9, 0.0, 1.0,
            #         -0.88, 0.82, 1.0, 0.0,
            #         -0.88, 0.9, 1.0, 1.0
            #     ], dtype=np.float32))
            #     text_surface = self.font.render(txt, True, Text.TXT_COLORS["white"], Text.TXT_COLORS["black"] if background else Text.TXT_COLORS["transparent"])
            #     self.textures.append(ctx.texture((text_surface.get_width(), text_surface.get_height()), 4, pg.image.tostring(text_surface, "RGBA", True)))
            # elif tidx == len(self.text)-1: #date
            #     self.vertices.append(np.array([
            #         -1.0, -1.0, 0.0, 0.0,
            #         -1.0, -0.95, 0.0, 1.0,
            #         -0.85, -1.0, 1.0, 0.0,
            #         -0.85, -0.95, 1.0, 1.0
            #     ], dtype=np.float32))
            #     text_surface = self.font.render(txt, True, Text.TXT_COLORS["white"], Text.TXT_COLORS["black"] if background else Text.TXT_COLORS["transparent"])
            #     self.textures.append(ctx.texture((text_surface.get_width(), text_surface.get_height()), 4, pg.image.tostring(text_surface, "RGBA", True)))
            # else: #team sizes
            #     # self.vertices.append(np.array([
            #     #     -1.08+tidx/10, 0.82, 0.0, 0.0,
            #     #     -1.08+tidx/10, 0.9, 0.0, 1.0,
            #     #     -1.03+tidx/10, 0.82, 1.0, 0.0,
            #     #     -1.03+tidx/10, 0.9, 1.0, 1.0
            #     # ], dtype=np.float32))
            #     inc = 0.08
            #     self.vertices.append(np.array([
            #         -0.88+(tidx-2)*inc, 0.82, 0.0, 0.0,
            #         -0.88+(tidx-2)*inc, 0.9, 0.0, 1.0,
            #         -0.83+(tidx-2)*inc, 0.82, 1.0, 0.0,
            #         -0.83+(tidx-2)*inc, 0.9, 1.0, 1.0
            #     ], dtype=np.float32))
            #     color = txt.split()[0]
            #     text_surface = self.font.render(txt.split()[1], True, Text.TXT_COLORS[color], Text.TXT_COLORS["black"] if background else Text.TXT_COLORS["transparent"])
            #     self.textures.append(ctx.texture((text_surface.get_width(), text_surface.get_height()), 4, pg.image.tostring(text_surface, "RGBA", True)))
            else:
                self.vertices.append(np.array([
                    -1.0, 0.82, 0.0, 0.0,
                    -1.0, 0.9, 0.0, 1.0,
                    -0.8, 0.82, 1.0, 0.0,
                    -0.8, 0.9, 1.0, 1.0
                ], dtype=np.float32))
                text_surface = self.font.render(txt, True, Text.TXT_COLORS["white"], Text.TXT_COLORS["black"] if background else Text.TXT_COLORS["transparent"])
                self.textures.append(ctx.texture((text_surface.get_width(), text_surface.get_height()), 4, pg.image.tostring(text_surface, "RGBA", True)))

            self.vaos.append(ctx.simple_vertex_array(self.prog, ctx.buffer(self.vertices[tidx].tobytes()), 'in_vert', 'in_texcoord'))


class Viewer:
    def __init__(self, width, height, caption=None,fps=30, headless=False, samples=4):
        """
        samples: anti-aliasing (MSAA) samples per pixel, 0 = anti-aliasing off.
        Values are rounded down to what the driver supports (power of two, at most ctx.max_samples).
        The window itself is created without MSAA (not every display server offers MSAA windows, e.g. WSLg),
        instead the scene is drawn into an offscreen MSAA framebuffer and copied to the window.
        """
        self.transform = Transform()
        self.width = width
        self.height = height
        self.fps = fps

        if headless:
            self.ctx = mgl.create_standalone_context()
            self.display = None
            self.clock = None
        else:
            pg.init()
            pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
            pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
            self.display = pg.display.set_mode((width, height), pg.DOUBLEBUF | pg.OPENGL)
            self.ctx = mgl.create_context()
            self.clock = pg.time.Clock()
            if caption:
                pg.display.set_caption(caption)

        # drivers only accept powers of two up to max_samples (e.g. 2 or 4), so round down
        samples = min(int(samples), self.ctx.max_samples)
        self.samples = 1 << (samples.bit_length() - 1) if samples > 1 else 0
        try:
            self.scene_fbo = self._create_scene_fbo(width, height, self.samples)
        except mgl.Error as e:
            warnings.warn(f"MSAA with {self.samples} samples not supported, rendering without anti-aliasing ({e})")
            self.samples = 0
            self.scene_fbo = self._create_scene_fbo(width, height, 0)
        # Final image without depth buffer: copied to the window and read by get_frame().
        # Copying to the window only works from a framebuffer without depth, since its depth format differs from the window's.
        self.fbo = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer((width, height))])

        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)
        
        self.projection = perspective(90, width/height, 0.1, 1000.)
        
        self.prog = load_shader(self.ctx, "simple.vert", "simple.frag")
        self.prog['projection'] = tuple(np.hstack(self.projection.T))
        self.prog['lightDir'] = .6, -.8, 1.0

        self.unlit = load_shader(self.ctx, "simple.vert", "unlit.frag")
        self.unlit['projection'] = tuple(np.hstack(self.projection.T))
        self.set_view()

        self.objects = []

    
    def run(self):
        running = True
        t = 0
        while running:
            pg.event.pump()
            camera_speed = .3
            camera_rot = np.pi/36
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
            
            self.callback()

            self.render()
            self.clock.tick(self.fps)
            t += 1
            
            #self.scene.root.children[0].transform.rotation = Quaternion.from_euler(0, t*np.pi/self.fps, 0)

        self.close()
    
    def callback(self):
        pass

    def set_view(self, x=None, y=None, z=None, rotation=None):
        if x is not None:
            self.transform.x = x
        if y is not None:
            self.transform.y = y
        if z is not None:
            self.transform.z = z
        if rotation is not None:
            self.transform.rotation = rotation
        self.prog['view'] = tuple(np.hstack(self.transform.inv_matrix.T))
        self.unlit['view'] = tuple(np.hstack(self.transform.inv_matrix.T))

    def _create_scene_fbo(self, width, height, samples):
        return self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer((width, height), samples=samples)],
            depth_attachment=self.ctx.depth_renderbuffer((width, height), samples=samples),
        )

    def get_frame(self):
        data = self.fbo.read()
        return np.array(bytearray(data)).reshape(self.height, self.width,3)[-1::-1,:,:]

    def render(self):
        if self.display is not None:
            pg.event.pump()

        self.scene_fbo.use()
        self.ctx.clear(0.5, 0.5, 0.5, 1.0)

        for obj in self.objects:
            obj.render()

        # resolves the MSAA samples (if any) into the final image
        self.ctx.copy_framebuffer(self.fbo, self.scene_fbo)
        if self.display is not None:
            self.ctx.copy_framebuffer(self.ctx.screen, self.fbo)
            pg.display.flip()
    
    def close(self):
        pg.quit()

# if __name__ == "__main__":
#     from numpy.linalg import inv
#     trans = Transform()
#     trans.position = -2,2,3
#     trans.rotation = Quaternion.from_euler(-.5,-.2,.3)
#     trans.scale = 5.0

#     print(np.sum(np.abs(trans.inv_matrix) - np.abs(inv(trans.matrix))))