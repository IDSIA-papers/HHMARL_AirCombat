import datetime
import numpy as np
from rendering.render import Viewer, load_mesh, RenderObject, Grid, Text
from rendering.quaternion import Quaternion

class Scene:
    SCALE = 1e-3
    RADIUS = 6.3781e6 #earth radius

    def __init__(self, env):
        self.viewer = Viewer(1280, 720)
        self.env = env
        self.objects = {}
        self.objects_dead_pos = {}
        self._load_objects()

    def render(self, viewpoint:int=0):
        """
        Update position and attitude in rendering scene.
        Aircraft with cannon is modeled as one object, so whenever an aircraft is shooting with cannon, we replace its object. 
        """
        self.viewer.objects.clear()
        curr_objects = [] #currently active objects

        for aid, agent in self.env.agents.items():
            if agent.alive:
                if agent.cannon.alive:
                    curr_object = self.objects[aid][1] #aircraft_cannon
                else:
                    curr_object = self.objects[aid][0] #aircraft
            # else:
            #     curr_object = self.objects[aid][2] #dead_cone

            state = []
            if agent.alive:
                for prop in agent.mission.render_var:
                    state.append(agent.get_property_value(prop))
                curr_object.transform.z = state[0] * Scene.SCALE * Scene.RADIUS
                curr_object.transform.x = -state[1] * Scene.SCALE * Scene.RADIUS
                curr_object.transform.y = state[2] * Scene.SCALE
                rot = Quaternion.from_euler(*state[3:])
                rot = Quaternion(rot.w, -rot.y, -rot.z, rot.x)
                curr_object.transform.rotation = rot
                self.objects_dead_pos[aid] = [curr_object.transform.x, curr_object.transform.y, curr_object.transform.z]
            
                curr_objects.append(curr_object)

        # concatenate active objects and grid layer
        self.viewer.objects.extend(curr_objects + [self.grid] + [self.text])
        self._render_objects(view=viewpoint)

    def _render_objects(self, view=0):
        """
        Viewpoints:
            0: top 45° down 
            1: top 90° down
            2: right 0° left
            3: from behind to front
        """
        if view==0:
            # x = left/right, y=height/depth, z=up/down
            view_point = np.array([0.,  30.,  -30])
            # x = up/down, y = left/right, z = rotate
            view_angle = Quaternion.from_euler(0.9, 0, 0, mode=1)
        elif view==1:
            view_point = np.array([0.,  30.,  0])
            view_angle = Quaternion.from_euler(1.55, 0, 0, mode=1)
        elif view==2:
            view_point = np.array([-20.,  5.,  0])
            view_angle = Quaternion.from_euler(0, 1.55, 0, mode=1)
        elif view==3:
            view_point = np.array([0.,  6.,  -20])
            view_angle = Quaternion.from_euler(0.2, 0, 0, mode=1)
        self.viewer.set_view(*view_point, rotation=view_angle)
        self.viewer.render()
        
    def _load_objects(self):
        """
        Load all aircraft and grid into scene. 
        """
        for aid, agent in self.env.agents.items():
            # aircraft without shooting cannon
            f16 = load_mesh(self.viewer.ctx, self.viewer.prog, "f16.obj")
            aircraft = RenderObject(f16)
            aircraft.transform.scale = 1/7
            aircraft.color = agent.color_code #e.g. [1,0,0]

            # aircraft with shooting cannon
            f16_cone = load_mesh(self.viewer.ctx, self.viewer.prog, "f16_cone.obj")
            aircraft_cone = RenderObject(f16_cone)
            aircraft_cone.transform.scale = 1/7
            aircraft_cone.color = agent.color_code

            # cone at position where agent died
            dead_cone_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, "cone.obj")
            dead_cone = RenderObject(dead_cone_mesh)
            dead_cone.transform.scale = Scene.SCALE * 300
            dead_cone.color = agent.color_code

            self.objects[aid] = [aircraft, aircraft_cone, dead_cone]

        self.grid = Grid(self.viewer.ctx, self.viewer.unlit, getattr(self.env.config, 'map_size', 100), 1., True)
        self.text = Text(self.viewer.ctx, self._load_text())
        #self.viewer.objects.extend(self.objects)

    def _load_text(self):
        """
        Write combat configuration text + date of today.
        """
        txt_objects = ["Combat " + '-vs-'.join(str(len(v)) for v in self.env.teams.values()), str(datetime.date.today())]
        return txt_objects