from math import cos,sin,tan,pi
import numpy as np

def rotate_x(angle):
    R = np.mat([[ 1,         0,           0],
                 [ 0, cos(angle), -sin(angle) ],
                 [ 0, sin(angle),  cos(angle) ]])
    return R

def rotate_y(angle):
    R = np.mat([[ cos(angle), 0,  sin(angle) ],
                 [ 0,         1,          0 ],
                 [-sin(angle), 0,  cos(angle) ]])
    return R

def rotate_z(angle):
    R = np.mat([[ cos(angle), -sin(angle), 0 ],
                 [ sin(angle),  cos(angle), 0 ],
                 [         0,          0, 1 ]])
    return R


class Converter():

    def __init__(self,camera_params):
        self._camera_params = camera_params
        self._intrinsic_matrix = self._get_intrinsic_matrix(self._camera_params)
        self.inv_extrinsic_matrix = None

    def _get_intrinsic_matrix(self,camera_params):

        # Calculate Inverse Intrinsic Matrix for both depth cameras
        f = camera_params["width"] /(2 * tan(camera_params["fov"] * pi / 360))
        Center_X = camera_params["width"] / 2.0
        Center_Y = camera_params["height"] / 2.0

        intrinsic_matrix = np.array([[f, 0, Center_X],
                                    [0, f, Center_Y],
                                    [0, 0, 1]])

        #return np.linalg.inv(intrinsic_matrix)
        return intrinsic_matrix

    # suppose camera orientation and position are 0,0,0 and camera_x,0,camera_z
    def _set_extrinsic_matrix(self,x,y,z,yaw):
        """
        HOW TO COMPUTE
        x   = measurements.player_measurements.transform.location.x + cam_x_pos
        y   = measurements.player_measurements.transform.location.y + cam_y_pos
        z   = cam_z_pos
        yaw = math.radians(measurements.player_measurements.transform.rotation.yaw)

        """

        extrinsic_matrix = np.zeros((4,4))
        R = np.dot(rotate_z(yaw + 90 * pi / 180),rotate_x(90 * pi / 180))

        # R = np.dot(rotate_x(0 - 90* pi / 180), rotate_y(-yaw + 90* pi / 180))
        # R = np.dot(R, rotate_z((0) + 90* pi / 180))

        extrinsic_matrix[:3,:3] = R
        extrinsic_matrix[:,-1]  = [x, y, z, 1]
        self.inv_extrinsic_matrix = extrinsic_matrix
    
    def convert_to_3D(self,pixel,pixel_depth,ego_x,ego_y,ego_yaw):
            """
            pixel should be [x,y,1]
            pixel_depth = depth_data[y1][x1]
            """
            # Projection Pixel to Image Frame
            inv_intrinsic_matrix = np.linalg.inv(self._intrinsic_matrix)
            camera_frame_point = np.dot(inv_intrinsic_matrix, pixel) * pixel_depth
            camera_frame_point = np.reshape(camera_frame_point,(3,1))
            
            world_frame_point = np.zeros((4,1))
            world_frame_point[:3] = camera_frame_point
            world_frame_point[-1] = 1

            sign_x = 1
            sign_y = 1

            # check if that's improve performance or not
            if ego_yaw>= 0 and ego_yaw<=pi/2:
                pass
            elif ego_yaw>pi/2 and ego_yaw<=pi:
                sign_x = -1
            elif ego_yaw> -pi and ego_yaw<= -pi/2:
                sign_x = -1
                sign_y = -1
            elif ego_yaw> -pi/2 and ego_yaw < 0:
                sign_y = -1

            # x   = ego_x + cos(ego_yaw)*self._camera_params["x"]
            # y   = ego_y 
            # z   = self._camera_params["z"]

            x   = ego_x + sign_x*self._camera_params["x"]
            y   = ego_y + sign_y*self._camera_params["y"]
            z   = self._camera_params["z"]
            yaw = ego_yaw
            self._set_extrinsic_matrix(x,y,z,yaw)
            world_frame_point = np.dot(self.inv_extrinsic_matrix, world_frame_point)
            world_frame_point = world_frame_point[:3]

            return world_frame_point

            
if __name__ == "__main__":
    camera_parameters = {}
    camera_parameters['x'] = 1.8 
    camera_parameters['y'] = 0.0
    camera_parameters['z'] = 1.3 
    camera_parameters['pitch'] = 0.0 
    camera_parameters['roll'] = 0.0
    camera_parameters['yaw'] = 0.0
    camera_parameters['width'] = 200 
    camera_parameters['height'] = 200 
    camera_parameters['fov'] = 90

    ego_x = 200
    ego_y = 200
    ego_yaw = 0

    depth = 5
    pixel = [100,100,1]
    c = Converter(camera_parameters)
    print(c.convert_to_3D(pixel,depth,ego_x,ego_y,ego_yaw))


