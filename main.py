#!/usr/bin/env python3
from __future__ import print_function
from __future__ import division

# System level imports
import sys
import os
import argparse
import logging
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import index
import controller2d
#import controller2d_AR as controller2d  
import configparser 
import local_planner
import behavioural_planner
import cv2
from math import sin, cos, pi, tan, sqrt
from utils import compute_middle_point
from agents import Agent
from sidewalk import point_in_sidewalks
from converter1 import Converter

import os

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import live_plotter as lv   # Custom live plotting library
from carla            import sensor
from carla.client     import make_carla_client, VehicleControl
from carla.settings   import CarlaSettings
from carla.tcp        import TCPConnectionError
from carla.controller import utils
from carla.sensor import Camera
from carla.image_converter import labels_to_array, depth_to_array, to_bgra_array
from carla.planner.city_track import CityTrack

from AVD_BP.carla_object_detector.carla_yolov3_model.OD import load_model,predict,postprocess
from AVD_BP.carla_object_detector.carla_yolov3_model.config import VEHICLE_TAG,PERSON_TAG

SERVER_HOST = "193.205.163.183"
SERVER_PORT = 6018

LOCAL_HOST = "localhost"
LOCAL_PORT = 2000

SIMULATION_PERFECT = False

###############################################################################
# CONFIGURABLE PARAMENTERS DURING EXAM
###############################################################################
PLAYER_START_INDEX = 15  #20 #89 #148   #91        #  spawn index for player
DESTINATION_INDEX = 139 #40# 133 #61   #142      # Setting a Destination HERE
NUM_PEDESTRIANS        = 500     # total number of pedestrians to spawn
NUM_VEHICLES           = 500        # total number of vehicles to spawn
SEED_PEDESTRIANS       = 0     # seed for pedestrian spawn randomizer
SEED_VEHICLES          = 1     # seed for vehicle spawn randomizer
###############################################################################

ITER_FOR_SIM_TIMESTEP  = 10     # no. iterations to compute approx sim timestep
WAIT_TIME_BEFORE_START = 1.00   # game seconds (time before controller start)
TOTAL_RUN_TIME         = 5000.00 # game seconds (total runtime before sim end)
TOTAL_FRAME_BUFFER     = 300    # number of frames to buffer after total runtime
CLIENT_WAIT_TIME       = 3      # wait time for client before starting episode
                                # used to make sure the server loads
                                # consistently

DESIRED_SPEED = 5.0

WINDOWS_OS = os.name == 'nt'

WEATHERID = {
    "DEFAULT": 0,
    "CLEARNOON": 1,
    "CLOUDYNOON": 2,
    "WETNOON": 3,
    "WETCLOUDYNOON": 4,
    "MIDRAINYNOON": 5,
    "HARDRAINNOON": 6,
    "SOFTRAINNOON": 7,
    "CLEARSUNSET": 8,
    "CLOUDYSUNSET": 9,
    "WETSUNSET": 10,
    "WETCLOUDYSUNSET": 11,
    "MIDRAINSUNSET": 12,
    "HARDRAINSUNSET": 13,
    "SOFTRAINSUNSET": 14,
}
SIMWEATHER = WEATHERID["CLEARNOON"]     # set simulation weather

FIGSIZE_X_INCHES   = 8      # x figure size of feedback in inches
FIGSIZE_Y_INCHES   = 8      # y figure size of feedback in inches
PLOT_LEFT          = 0.1    # in fractions of figure width and height
PLOT_BOT           = 0.1    
PLOT_WIDTH         = 0.8
PLOT_HEIGHT        = 0.8

DIST_THRESHOLD_TO_LAST_WAYPOINT = 2.0  # some distance from last position before
                                       # simulation ends

# Planning Constants
NUM_PATHS = 7
BP_LOOKAHEAD_BASE      = 16.0              # m
BP_LOOKAHEAD_TIME      = 1.0              # s
PATH_OFFSET            = 1.5              # m
CIRCLE_OFFSETS         = [-1.0, 1.0, 3.0] # m
CIRCLE_RADII           = [1.5, 1.5, 1.5]  # m
TIME_GAP               = 1.0              # s
PATH_SELECT_WEIGHT     = 10
A_MAX                  = 2.5              # m/s^2
SLOW_SPEED             = 2.0              # m/s
STOP_LINE_BUFFER       = 3.5              # m
LEAD_VEHICLE_LOOKAHEAD = 20.0             # m
LP_FREQUENCY_DIVISOR   = 2                # Frequency divisor to make the 
                                          # local planner operate at a lower
                                          # frequency than the controller
                                          # (which operates at the simulation
                                          # frequency). Must be a natural
                                          # number.

# Path interpolation parameters
INTERP_MAX_POINTS_PLOT    = 10   # number of points used for displaying
                                 # selected path
INTERP_DISTANCE_RES       = 0.01 # distance between interpolated points

# controller output directory
CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) +\
                           '/controller_output/'

AGENTS_CHECK_RADIUS = 30

# Camera parameters
camera_parameters = {}
camera_parameters['x'] = 1.8 
camera_parameters['y'] = 0.0
camera_parameters['z'] = 1.3
camera_parameters['pitch'] = 0.0 
camera_parameters['roll'] = 0.0
camera_parameters['yaw'] = 0.0
camera_parameters['width'] = 224#200 
camera_parameters['height'] = 224#200 
camera_parameters['fov'] = 90

camera_parameters_bis = {}
camera_parameters_bis['x'] = 1.8 
camera_parameters_bis['y'] = 0.0
camera_parameters_bis['z'] = 1.3
camera_parameters_bis['pitch'] = 0.0 
camera_parameters_bis['roll'] = 0.0
camera_parameters_bis['yaw'] = 0.0
camera_parameters_bis['width'] = 224#200 
camera_parameters_bis['height'] = 224#200 
camera_parameters_bis['fov'] = 120

camera_parameters_view = {}
camera_parameters_view['x'] = -5.0
camera_parameters_view['y'] = 0.0
camera_parameters_view['z'] = 2.5
camera_parameters_view['pitch'] = -15.0
camera_parameters_view['roll'] = 0.0
camera_parameters_view['yaw'] = 0.0
camera_parameters_view['width'] = 500 
camera_parameters_view['height'] = 500
camera_parameters_view['fov'] = 90

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


def find_pedestrians_and_vehicles_from_camera(net, camera_data, seg_data, depth_data, current_x, current_y, current_z, current_yaw, camera_parameters, bis=False): 
    """
        Args: 
        - net: rete addestrata a fare object detection su immagini
        - camera_data: dati della telecamera
        - seg_data: dati della telecamera di semantic segmentation
        - depth_data: dati dalla telecamera di profondità
        Returns:
        - world_frame_pedestrians: lista di coordinate (x,y) dei pedoni rilevati dalla telecamera nel mondo reale
        - world_frame_vehicles: come sopra ma per i veicoli
    """
    converter = Converter(camera_parameters)
    
    ###################################
    # GET BBs

    bb = predict(net,camera_data)
    camera_data, bb_dict= postprocess(camera_data,bb)
    if bis:
        cv2.imshow("Detection box bis",camera_data)
    else:
        cv2.imshow("Detection box",camera_data)
    cv2.waitKey(10)
    
    
    #bbs vehicle and pedestrian
    ## bb_p and bb_v are lists like [[(x,y),width,height]]
    # NOTE to access to a specific pixel from bb x,y -> camera_data[y,x] 
    #list of pedestrians bounding boxes
    bb_p = bb_dict[PERSON_TAG]
    
    # list of bounding boxis

    bb_v = bb_dict[VEHICLE_TAG]


    ###################################
    # MARK PEDESTRIAN BB ON SIDEWAYS
    
    # only pedestrian bb

    # found point in the middle of bb vertex like X, x1 refer to (x,y) from one bb in bb_p
    #   x1--------x2
    #    |        | 
    #   x3---X----x4
    # 
    # if X is on sidewalk (or x3 or x4) mark this pedestrian as on sidewalk
    # in this section for each pedestrian bb check if point X is on sidewalk
    
    # USE FUNCTION : point_in_sidewalks(semSeg_data, point) NOTE: point must be provided as (y,x)

    
    count=0
    sidewalk= {} #contains only the X on sidewalk, True if X is on sidewalk otherwise False
    for bb in bb_p:
        middle_point = compute_middle_point(bb[0][0], bb[0][1], bb[1], bb[2])
        on_sidewalk = point_in_sidewalks(seg_data, middle_point)
        sidewalk[count] = on_sidewalk

        count+=1
        

    ###################################
    # FOR EACH BB WE CAN CHOOSE X POINT DESCIBED IN PREVIUS SECTION TO GET VEHICLES POSITION
    # IN 3D WORLD COORDINATE FRAME
    # USING DEPTH CAMERA GET PEDESTRIAN BB AND VEHICLE BB IN WORLD COORDINATES FRAME


    # USE this to convert a pixel in 3D  pixel should be [x,y,1] pixel_depth = depth_data[y1][x1]
    #converter.convert_to_3D(pixel,pixel_depth,current_x,current_y,current_yaw)

    world_frame_vehicles = [] #list of tuples of converted pixel in the world
    for vehicle in bb_v:
        middle_point = compute_middle_point(vehicle[0][0], vehicle[0][1], vehicle[1], vehicle[2])
        middle_point = (min(middle_point[0],camera_parameters['height']-1), min(middle_point[1], camera_parameters['width']-1))
        #pixel = [middle_point[0], middle_point[1], 1]
        pixel = [middle_point[0], middle_point[1]]
        pixel_depth = depth_data[middle_point[1], middle_point[0]]*1000
        # world_frame_point= converter.convert_to_3D(pixel, pixel_depth, current_x, current_y,current_yaw)
        world_frame_point= converter.convert(pixel, pixel_depth, current_x, current_y,current_z,current_yaw)
        
        world_frame_vehicles.append(world_frame_point)

    world_frame_pedestrians = [] #list of tuples of converted pixel in the world
    for pedestrian in bb_p:
        middle_point = compute_middle_point(pedestrian[0][0], pedestrian[0][1], pedestrian[1], pedestrian[2])
        middle_point = (min(middle_point[0],camera_parameters['height']-1), min(middle_point[1], camera_parameters['width']-1))
        # pixel = [middle_point[0], middle_point[1], 1]
        pixel = [middle_point[0], middle_point[1]]

        pixel_depth = depth_data[middle_point[1], middle_point[0]]*1000
        # world_frame_point= converter.convert_to_3D(pixel, pixel_depth, current_x, current_y,current_yaw)
        world_frame_point= converter.convert(pixel, pixel_depth, current_x, current_y,current_z,current_yaw)
        
        world_frame_pedestrians.append(world_frame_point)
    
    # # print("[EGO LOCATION]", current_x,current_y)
    # # TESTING
    # # print("[VEHICLES DETECTED FROM CAMERA]: ")
    # if bis:
    #     print("CAMERA_BIS")
    # else:
    #     print("CAMERA")
    # for v in world_frame_vehicles:
    #     print("[VEHICLES DETECTED FROM CAMERA]",v[0],v[1])
    # for i,p in enumerate(world_frame_pedestrians):
    #     print("[PEDESTRIANS DETECTED FROM CAMERA]",p[0],p[1],"sidewalk:", sidewalk[i])
    #     #print(f"{p}, sidewalk: {sidewalk[i]}")

    return world_frame_vehicles, world_frame_pedestrians, sidewalk

# Transform the obstacle with its boundary point in the global frame
# bounding_box.transform.location, bounding_box.extent ,bounding_box.transform.rotation
def obstacle_to_world(location, dimensions, orientation):
    box_pts = []

    x = location.x
    y = location.y
    z = location.z

    yaw = orientation.yaw * pi / 180

    xrad = dimensions.x
    yrad = dimensions.y
    zrad = dimensions.z

    # Border points in the obstacle frame
    cpos = np.array([
            [-xrad, -xrad, -xrad, 0,    xrad, xrad, xrad,  0    ],
            [-yrad, 0,     yrad,  yrad, yrad, 0,    -yrad, -yrad]])
    
    # Rotation of the obstacle
    rotyaw = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]])
    
    # Location of the obstacle in the world frame
    cpos_shift = np.array([
            [x, x, x, x, x, x, x, x],
            [y, y, y, y, y, y, y, y]])
    
    cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)

    for j in range(cpos.shape[1]):
        box_pts.append([cpos[0,j], cpos[1,j]])
    
    return box_pts

def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need.
    """
    settings = CarlaSettings()
    
    # There is no need for non-agent info requests if there are no pedestrians
    # or vehicles.
    get_non_player_agents_info = False
    if (NUM_PEDESTRIANS > 0 or NUM_VEHICLES > 0):
        get_non_player_agents_info = True

    # Base level settings
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=get_non_player_agents_info, 
        NumberOfVehicles=NUM_VEHICLES,
        NumberOfPedestrians=NUM_PEDESTRIANS,
        SeedVehicles=SEED_VEHICLES,
        SeedPedestrians=SEED_PEDESTRIANS,
        WeatherId=SIMWEATHER,
        QualityLevel=args.quality_level)

    # Common cameras settings
    cam_height = camera_parameters['z'] 
    cam_x_pos = camera_parameters['x']
    cam_y_pos = camera_parameters['y']
    camera_pitch = camera_parameters['pitch']
    camera_roll = camera_parameters['roll']
    camera_yaw = camera_parameters['yaw']
    camera_width = camera_parameters['width']
    camera_height = camera_parameters['height']
    camera_fov = camera_parameters['fov']

    cam_height_bis = camera_parameters_bis['z'] 
    cam_x_pos_bis = camera_parameters_bis['x']
    cam_y_pos_bis = camera_parameters_bis['y']
    camera_pitch_bis = camera_parameters_bis['pitch']
    camera_roll_bis = camera_parameters_bis['roll']
    camera_yaw_bis = camera_parameters_bis['yaw']
    camera_width_bis = camera_parameters_bis['width']
    camera_height_bis = camera_parameters_bis['height']
    camera_fov_bis = camera_parameters_bis['fov']

    # Declare here your sensors
    camera0 = Camera("CameraRGB")
    camera0.set_image_size(camera_width, camera_height)
    camera0.set(FOV=camera_fov)
    camera0.set_position(cam_x_pos, cam_y_pos, cam_height)
    camera0.set_rotation(camera_pitch, camera_roll, camera_yaw)

    camera0bis = Camera("CameraRGBbis")
    camera0bis.set_image_size(camera_width_bis, camera_height_bis)
    camera0bis.set(FOV=camera_fov_bis)
    camera0bis.set_position(cam_x_pos_bis, cam_y_pos_bis, cam_height_bis)
    camera0bis.set_rotation(camera_pitch_bis, camera_roll_bis, camera_yaw_bis)

    camera1 = Camera("CameraSemSeg", PostProcessing="SemanticSegmentation")
    camera1.set_image_size(camera_width, camera_height)
    camera1.set(FOV=camera_fov)
    camera1.set_position(cam_x_pos, cam_y_pos, cam_height)
    camera1.set_rotation(camera_pitch, camera_roll, camera_yaw)

    camera1bis = Camera("CameraSemSegbis", PostProcessing="SemanticSegmentation")
    camera1bis.set_image_size(camera_width_bis, camera_height_bis)
    camera1bis.set(FOV=camera_fov_bis)
    camera1bis.set_position(cam_x_pos_bis, cam_y_pos_bis, cam_height_bis)
    camera1bis.set_rotation(camera_pitch_bis, camera_roll_bis, camera_yaw_bis)

    camera2 = Camera("CameraDepth", PostProcessing="Depth")
    camera2.set_image_size(camera_width, camera_height)
    camera2.set(FOV=camera_fov)
    camera2.set_position(cam_x_pos, cam_y_pos, cam_height)
    camera2.set_rotation(camera_pitch, camera_roll, camera_yaw)

    camera2bis = Camera("CameraDepthbis", PostProcessing="Depth")
    camera2bis.set_image_size(camera_width_bis, camera_height_bis)
    camera2bis.set(FOV=camera_fov_bis)
    camera2bis.set_position(cam_x_pos_bis, cam_y_pos_bis, cam_height_bis)
    camera2bis.set_rotation(camera_pitch_bis, camera_roll_bis, camera_yaw_bis)

    settings.add_sensor(camera0)
    settings.add_sensor(camera0bis)
    settings.add_sensor(camera1)
    settings.add_sensor(camera1bis)
    settings.add_sensor(camera2)
    settings.add_sensor(camera2bis)
        
    if not args.local:
        # Common cameras settings
        cam_height = camera_parameters_view['z'] 
        cam_x_pos = camera_parameters_view['x']
        cam_y_pos = camera_parameters_view['y']
        camera_pitch = camera_parameters_view['pitch']
        camera_roll = camera_parameters_view['roll']
        camera_yaw = camera_parameters_view['yaw']
        camera_width = camera_parameters_view['width']
        camera_height = camera_parameters_view['height']
        camera_fov = camera_parameters_view['fov']


        # Declare here your sensors
        camera3 = Camera("CameraRGBView")
        camera3.set_image_size(camera_width, camera_height)
        camera3.set(FOV=camera_fov)
        camera3.set_position(cam_x_pos, cam_y_pos, cam_height)
        camera3.set_rotation(camera_pitch, camera_roll, camera_yaw)

    
        settings.add_sensor(camera3)

    return settings

class Timer(object):
    """ Timer Class
    
    The steps are used to calculate FPS, while the lap or seconds since lap is
    used to compute elapsed time.
    """
    def __init__(self, period):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()
        self._period_for_lap = period

    def tick(self):
        self.step += 1

    def has_exceeded_lap_period(self):
        if self.elapsed_seconds_since_lap() >= self._period_for_lap:
            return True
        else:
            return False

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) /\
                     self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time

def get_current_pose(measurement):
    """Obtains current x,y,yaw pose from the client measurements
    
    Obtains the current x,y, and yaw pose from the client measurements.

    Args:
        measurement: The CARLA client measurements (from read_data())

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x   = measurement.player_measurements.transform.location.x
    y   = measurement.player_measurements.transform.location.y
    z   =  measurement.player_measurements.transform.location.z

    pitch = math.radians(measurement.player_measurements.transform.rotation.pitch)
    roll = math.radians(measurement.player_measurements.transform.rotation.roll)
    yaw = math.radians(measurement.player_measurements.transform.rotation.yaw)

    return (x, y, z, pitch, roll, yaw)

def get_start_pos(scene):
    """Obtains player start x,y, yaw pose from the scene
    
    Obtains the player x,y, and yaw pose from the scene.

    Args:
        scene: The CARLA scene object

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x = scene.player_start_spots[0].location.x
    y = scene.player_start_spots[0].location.y
    yaw = math.radians(scene.player_start_spots[0].rotation.yaw)

    return (x, y, yaw)

def get_player_collided_flag(measurement, 
                             prev_collision_vehicles, 
                             prev_collision_pedestrians,
                             prev_collision_other):
    """Obtains collision flag from player. Check if any of the three collision
    metrics (vehicles, pedestrians, others) from the player are true, if so the
    player has collided to something.

    Note: From the CARLA documentation:

    "Collisions are not annotated if the vehicle is not moving (<1km/h) to avoid
    annotating undesired collision due to mistakes in the AI of non-player
    agents."
    """
    player_meas = measurement.player_measurements
    current_collision_vehicles = player_meas.collision_vehicles
    current_collision_pedestrians = player_meas.collision_pedestrians
    current_collision_other = player_meas.collision_other

    collided_vehicles = current_collision_vehicles > prev_collision_vehicles
    collided_pedestrians = current_collision_pedestrians > \
                           prev_collision_pedestrians
    collided_other = current_collision_other > prev_collision_other

    return (collided_vehicles or collided_pedestrians or collided_other,
            current_collision_vehicles,
            current_collision_pedestrians,
            current_collision_other)

def send_control_command(client, throttle, steer, brake, 
                         hand_brake=False, reverse=False):
    """Send control command to CARLA client.
    
    Send control command to CARLA client.

    Args:
        client: The CARLA client object
        throttle: Throttle command for the sim car [0, 1]
        steer: Steer command for the sim car [-1, 1]
        brake: Brake command for the sim car [0, 1]
        hand_brake: Whether the hand brake is engaged
        reverse: Whether the sim car is in the reverse gear
    """
    control = VehicleControl()
    # Clamp all values within their limits
    steer = np.fmax(np.fmin(steer, 1.0), -1.0)
    throttle = np.fmax(np.fmin(throttle, 1.0), 0)
    brake = np.fmax(np.fmin(brake, 1.0), 0)

    control.steer = steer
    control.throttle = throttle
    control.brake = brake
    control.hand_brake = hand_brake
    control.reverse = reverse
    client.send_control(control)

def create_controller_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def store_trajectory_plot(graph, fname):
    """ Store the resulting plot.
    """
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)

    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
    graph.savefig(file_name)

def write_trajectory_file(x_list, y_list, v_list, t_list, collided_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'trajectory.txt')

    with open(file_name, 'w') as trajectory_file: 
        for i in range(len(x_list)):
            trajectory_file.write('%3.3f, %3.3f, %2.3f, %6.3f %r\n' %\
                                  (x_list[i], y_list[i], v_list[i], t_list[i],
                                   collided_list[i]))

def write_collisioncount_file(collided_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'collision_count.txt')

    with open(file_name, 'w') as collision_file: 
        collision_file.write(str(sum(collided_list)))

def make_correction(waypoint,previuos_waypoint,desired_speed):
    dx = waypoint[0] - previuos_waypoint[0]
    dy = waypoint[1] - previuos_waypoint[1]

    if dx < 0:
        moveY = -1.5
    elif dx > 0:
        moveY = 1.5
    else:
        moveY = 0

    if dy < 0:
        moveX = 1.5
    elif dy > 0:
        moveX = -1.5
    else:
        moveX = 0
    
    waypoint_on_lane = waypoint
    waypoint_on_lane[0] += moveX
    waypoint_on_lane[1] += moveY
    waypoint_on_lane[2] = desired_speed

    return waypoint_on_lane


def found_nearest_object(position,objects_position,objects_just_assoicated):
    """
        Given the list of objects position found the index of the object position
        nearest to the given position.
        All indices just used are provided in objects_just_associated list
    """
    THRESHOLD_DISTANCE = 3
    min_index = None
    min_dist = math.inf

    for i, object_position in enumerate(objects_position): #from camera0
        x_point, y_point = object_position[0][0], object_position[1][0] # prendere i dati dagli attributi di world_frame
        # print("NEAREST_FUNC ",x_point,y_point,"\n")
        dist = np.subtract(position,[x_point, y_point])
        norm = np.linalg.norm(dist)
        # an association is found
        if norm < min_dist and norm < THRESHOLD_DISTANCE and i not in objects_just_assoicated:
            # print("[NEAREST_FUNC] REALDATA ",x_point,y_point)
            # print("[NEAREST_FUNC] PERFECT DATA ",position,"\n")
            min_dist = norm
            min_index = i
    return min_index


def association_vehicle_pedestrian(perfect_data, real_data, real_data_bis, sidewalk=None, sidewalk_bis = None, pedestrian=False):
    # THRESHOLD_DISTANCE = 2.5
    THRESHOLD_SPEED = 0.15

    indices_associated = []
    data_to_consider = []
    indices_associated_bis = []
    vehicle_dict = {}
    
    # for each real data to associate to given detected data
    for d in perfect_data:
        x, y = d.get_position()
        
        # print("Pedestrian: ", pedestrian)
        min_index= found_nearest_object([x,y],real_data,indices_associated)
        min_index_bis = found_nearest_object([x,y],real_data_bis,indices_associated_bis)

        
        # real objcet index. 
        association_index = None

        # sidewalk for pedestrian association 
        sidewalk_to_consider = None
        pose = None
        #if a perfect object is associated to both real_data and real_data_bis we
        # decide to associate it to real_data object
        if min_index is None and min_index_bis != None:
            association_index = min_index_bis
            pose = real_data_bis[association_index]
            sidewalk_to_consider = sidewalk_bis
            indices_associated_bis.append(min_index_bis)
        elif min_index != None:
            association_index = min_index
            pose = real_data[association_index]
            sidewalk_to_consider = sidewalk
            indices_associated.append(min_index)


        # if not pedestrian and (min_index != None or min_index_bis != None):
        #     camera_used = "BIS" if min_index_bis != None else "0"
        #     print(f"ASSOCIATED VEHICLES FROM CAMERA {camera_used}: {pose.reshape(1,3)[:2]} to vehicle {d.get_position()}")
        # if an association is found
        if association_index is not None: 
            # pose = real_data[association_index]
            position = (pose[0][0],pose[1][0])
            # print("[MAIN] TEST line 693",type(position[0]),position[0],position[0][0])
            #position = d.get_position()
            yaw = d.get_orientation()
            bb = d.get_bounding_box()
            speed = d.get_speed()
            id = d.get_id()
            if not pedestrian:
                vehicle = Agent(id,position,bb,yaw,speed,"Vehicle")
                data_to_consider.append(vehicle)
                if not SIMULATION_PERFECT:
                    vehicle_dict[id] = vehicle
            else:
                # if the detected pedestrian is one sidewalk and its speed is less than THRESHOLD_SPEED
                # no association must be made
                if sidewalk_to_consider is not None and not(sidewalk_to_consider[association_index] and speed<THRESHOLD_SPEED):
                    data_to_consider.append(Agent(id,position,bb,yaw,speed,"Pedestrian"))

    
    return data_to_consider, vehicle_dict
                    
def agent_entering_management(current_agents,last_agents, entering,vehicles_dict = None):
    agents_to_consider = []
    MIN_ENTERING_FRAME = 2
    # entering pedestrian

    # STEP 1: update entering objects
    for current_agent in current_agents: # from each pedestrian in current frame
        id = current_agent.get_id()
        # this boolean var check if a pedestrain is just detected in the scene
        # if in next iteration it is not updated to True means that this object is 
        # an entering object
        check_existing = False
        for last_agent in last_agents:
            if id == last_agent.get_id(): # check if it just detected in the last frame
                check_existing = True
                agents_to_consider.append(current_agent)
                if vehicles_dict is not None:
                    vehicles_dict[id] = current_agent
                break 

        # if a match between the current and last frame is check_existing 
        #  so it is an entering object 
        if check_existing:
            if id in entering:
                entering[id][0]+=1
                entering[id][1] = current_agent # update location and speed 
            else:
                entering[id] = [1,current_agent]

    # STEP 2: for each entering object check if enough frame have passed from entering condition 

    entering_ids = list(entering.keys())
    for id in entering_ids:
        counter = entering[id][0]
        if counter == MIN_ENTERING_FRAME:
            agents_to_consider.append(current_agent)
            if vehicles_dict is not None:
                vehicles_dict[id] = current_agent
            # there is no need to flag this object as entering object because now it is a real object
            del entering[id]

    # STEP 3: delete all entering object that are not are detected in the current frame 
    # thats means that they were FALSE POSITIVE objects

    for id in entering_ids:
        # flag to detect wich object can maintains the entering conditions
        check_entering_condition = False
        for current_agent in current_agents:
            if id == current_agent.get_id():
                check_entering_condition = True
                break
        
        if not check_entering_condition:
            del entering[id]

    print("ENTERING INSIDE FUNCTION")
    print(entering)

    return agents_to_consider, entering, vehicles_dict

def agents_outgoing_managements(current_agents,last_agents, outgoing, vehicle_dict=None):
    agents_to_consider = []
    MAX_GHOST_FRAME = 5

    # STEP 1: update ghost object
    for last_agent in last_agents:
        id = last_agent.get_id()
        check_ghost = True
        for current_agent in current_agents:
            if id == current_agent.get_id():
                check_ghost = False
                break

        # update number of frame where this object is a ghost
        if check_ghost:
            if id in outgoing:
                outgoing[id][0]+=1
            else:
                outgoing[id] = [1, last_agent]
        # delete agents that are not ghost yet 
        else: 
            del outgoing[id]

    
    # STEP 2: check which object should be delete from ghost condition

    ids_ghost = list(outgoing.keys())

    for id in ids_ghost:
        if outgoing[id][0] < MAX_GHOST_FRAME:
            agent = outgoing[id][1]
            agents_to_consider.append(agent)
            if vehicle_dict is not None:
                vehicle_dict[id]=agent
        else:
            del outgoing[id] # if MAX_GHOST_FRAME are passed 


    return agents_to_consider, outgoing, vehicle_dict

def exec_waypoint_nav_demo(args, host, port):
    """ Executes waypoint navigation demo.
    """
    with make_carla_client(host, port) as client:
        print('Carla client connected.')

        settings = make_carla_settings(args)

        # Now we load these settings into the server. The server replies
        # with a scene description containing the available start spots for
        # the player. Here we can provide a CarlaSettings object or a
        # CarlaSettings.ini file as string.
        scene = client.load_settings(settings)

        # Refer to the player start folder in the WorldOutliner to see the 
        # player start information
        player_start = args.start   

        # Notify the server that we want to start the episode at the
        # player_start index. This function blocks until the server is ready
        # to start the episode.
        print('Starting new episode at %r...' % scene.map_name)
        client.start_episode(player_start)

        #############################################
        # Load Configurations
        #############################################

        # Load configuration file (options.cfg) and then parses for the various
        # options. Here we have two main options:
        # live_plotting and live_plotting_period, which controls whether
        # live plotting is enabled or how often the live plotter updates
        # during the simulation run.
        config = configparser.ConfigParser()
        config.read(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'options.cfg'))         
        demo_opt = config['Demo Parameters']

        # Get options
        enable_live_plot = demo_opt.get('live_plotting', 'true').capitalize()
        enable_live_plot = enable_live_plot == 'True'
        live_plot_period = float(demo_opt.get('live_plotting_period', 0))

        # Set options
        live_plot_timer = Timer(live_plot_period)
        
        # Settings Mission Planner
        mission_planner = CityTrack("Town01")

        #############################################
        # Determine simulation average timestep (and total frames)
        #############################################
        # Ensure at least one frame is used to compute average timestep
        num_iterations = ITER_FOR_SIM_TIMESTEP
        if (ITER_FOR_SIM_TIMESTEP < 1):
            num_iterations = 1

        # Gather current data from the CARLA server. This is used to get the
        # simulator starting game time. Note that we also need to
        # send a command back to the CARLA server because synchronous mode
        # is enabled.
        measurement_data, sensor_data = client.read_data()

        car_extent_x = measurement_data.player_measurements.bounding_box.extent.x
        car_extent_y = measurement_data.player_measurements.bounding_box.extent.y

        # get traffic light information
        traffic_lights = [] #[id, [x,y],yaw]
        for agent in measurement_data.non_player_agents:
            if agent.HasField("traffic_light"):
                traffic_lights.append([agent.id,
                agent.traffic_light.transform.location.x,agent.traffic_light.  transform.location.y,
                agent.traffic_light.transform.rotation.yaw,agent.traffic_light.state])
                


        sim_start_stamp = measurement_data.game_timestamp / 1000.0
        # Send a control command to proceed to next iteration.
        # This mainly applies for simulations that are in synchronous mode.
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        # Computes the average timestep based on several initial iterations
        sim_duration = 0
        for i in range(num_iterations):
            # Gather current data
            measurement_data, sensor_data = client.read_data()
            # Send a control command to proceed to next iteration
            send_control_command(client, throttle=0.0, steer=0, brake=1.0)
            # Last stamp
            if i == num_iterations - 1:
                sim_duration = measurement_data.game_timestamp / 1000.0 -\
                               sim_start_stamp

        # Outputs average simulation timestep and computes how many frames
        # will elapse before the simulation should end based on various
        # parameters that we set in the beginning.
        SIMULATION_TIME_STEP = sim_duration / float(num_iterations)
        print("SERVER SIMULATION STEP APPROXIMATION: " + \
              str(SIMULATION_TIME_STEP))
        TOTAL_EPISODE_FRAMES = int((TOTAL_RUN_TIME + WAIT_TIME_BEFORE_START) /\
                               SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER

        #############################################
        # Frame-by-Frame Iteration and Initialization
        #############################################
        # Store pose history starting from the start position
        measurement_data, sensor_data = client.read_data()            



        start_timestamp = measurement_data.game_timestamp / 1000.0
        start_x, start_y, start_z, start_pitch, start_roll, start_yaw = get_current_pose(measurement_data)
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        x_history     = [start_x]
        y_history     = [start_y]
        yaw_history   = [start_yaw]
        time_history  = [0]
        speed_history = [0]
        collided_flag_history = [False]  # assume player starts off non-collided

        #############################################
        # Settings Waypoints
        #############################################
        starting    = scene.player_start_spots[args.start]
        destination = scene.player_start_spots[args.dest]

        # Starting position is the current position
        # (x, y, z, pitch, roll, yaw)
        source_pos = [starting.location.x, starting.location.y, starting.location.z]
        source_ori = [starting.orientation.x, starting.orientation.y]
        source = mission_planner.project_node(source_pos)

        # Destination position
        destination_pos = [destination.location.x, destination.location.y, destination.location.z]
        destination_ori = [destination.orientation.x, destination.orientation.y]
        destination = mission_planner.project_node(destination_pos)

        waypoints = []
        waypoints_route = mission_planner.compute_route(source, source_ori, destination, destination_ori)
        desired_speed = DESIRED_SPEED
        turn_speed    = 2.5

        intersection_nodes = mission_planner.get_intersection_nodes()
        intersection_pair = []
        turn_cooldown = 0
        prev_x = False
        prev_y = False
        # Put waypoints in the lane
        previuos_waypoint = mission_planner._map.convert_to_world(waypoints_route[0])
        for i in range(1,len(waypoints_route)):
            point = waypoints_route[i]

            waypoint = mission_planner._map.convert_to_world(point)

            current_waypoint = make_correction(waypoint,previuos_waypoint,desired_speed)
            
            dx = current_waypoint[0] - previuos_waypoint[0]
            dy = current_waypoint[1] - previuos_waypoint[1]

            is_turn = ((prev_x and abs(dy) > 0.1) or (prev_y and abs(dx) > 0.1)) and not(abs(dx) > 0.1 and abs(dy) > 0.1)

            prev_x = abs(dx) > 0.1
            prev_y = abs(dy) > 0.1

            if point in intersection_nodes:                
                prev_start_intersection = mission_planner._map.convert_to_world(waypoints_route[i-2])
                center_intersection = mission_planner._map.convert_to_world(waypoints_route[i])

                start_intersection = mission_planner._map.convert_to_world(waypoints_route[i-1])
                end_intersection = mission_planner._map.convert_to_world(waypoints_route[i+1])

                start_intersection = make_correction(start_intersection,prev_start_intersection,turn_speed)
                end_intersection = make_correction(end_intersection,center_intersection,turn_speed)
                
                dx = start_intersection[0] - end_intersection[0]
                dy = start_intersection[1] - end_intersection[1]

                if abs(dx) > 0 and abs(dy) > 0:
                    intersection_pair.append((center_intersection,len(waypoints)))
                    waypoints[-1][2] = turn_speed
                    
                    middle_point = [(start_intersection[0] + end_intersection[0]) /2,  (start_intersection[1] + end_intersection[1]) /2]

                    centering = 0.75

                    middle_intersection = [(centering*middle_point[0] + (1-centering)*center_intersection[0]),  (centering*middle_point[1] + (1-centering)*center_intersection[1])]

                    # Point at intersection:
                    A = [[start_intersection[0], start_intersection[1], 1],
                         [end_intersection[0], end_intersection[1], 1],
                         [middle_intersection[0], middle_intersection[1], 1]]
                        
                    b = [-start_intersection[0]**2 - start_intersection[1]**2, 
                         -end_intersection[0]**2 - end_intersection[1]**2,
                         -middle_intersection[0]**2 - middle_intersection[1]**2]

                    coeffs = np.matmul(np.linalg.inv(A), b)

                    x = start_intersection[0]
                    
                    center_x = -coeffs[0]/2
                    center_y = -coeffs[1]/2

                    r = sqrt(center_x**2 + center_y**2 - coeffs[2])

                    theta_start = math.atan2((start_intersection[1] - center_y),(start_intersection[0] - center_x))
                    theta_end = math.atan2((end_intersection[1] - center_y),(end_intersection[0] - center_x))

                    theta = theta_start

                    start_to_end = 1 if theta_start < theta_end else -1

                    while (start_to_end==1 and theta < theta_end) or (start_to_end==-1 and theta > theta_end):
                        waypoint_on_lane = [0,0,0]

                        waypoint_on_lane[0] = center_x + r * cos(theta)
                        waypoint_on_lane[1] = center_y + r * sin(theta)
                        waypoint_on_lane[2] = turn_speed

                        waypoints.append(waypoint_on_lane)
                        theta += (abs(theta_end - theta_start) * start_to_end) / 10
                    
                    turn_cooldown = 4
            else:
                waypoint = mission_planner._map.convert_to_world(point)

                if turn_cooldown > 0:
                    target_speed = turn_speed
                    turn_cooldown -= 1
                else:
                    target_speed = desired_speed
                
                waypoint_on_lane = make_correction(waypoint,previuos_waypoint,target_speed)

                waypoints.append(waypoint_on_lane)

                previuos_waypoint = waypoint

        

        waypoints = np.array(waypoints)
        # print("[MAIN] n waypoints -> ", len(waypoints))
        # with open("waypoints.txt","w") as f:
        #     for x,y,v in waypoints:
        #         f.writelines(f"{x}, {y}, {v}\n")

        #############################################
        # Controller 2D Class Declaration
        #############################################
        # This is where we take the controller2d.py class
        # and apply it to the simulator
        controller = controller2d.Controller2D(waypoints)

        #############################################
        # Vehicle Trajectory Live Plotting Setup
        #############################################
        # Uses the live plotter to generate live feedback during the simulation
        # The two feedback includes the trajectory feedback and
        # the controller feedback (which includes the speed tracking).
        lp_traj = lv.LivePlotter(tk_title="Trajectory Trace")
        lp_1d = lv.LivePlotter(tk_title="Controls Feedback")

        ###
        # Add 2D position / trajectory plot
        ###
        trajectory_fig = lp_traj.plot_new_dynamic_2d_figure(
                title='Vehicle Trajectory',
                figsize=(FIGSIZE_X_INCHES, FIGSIZE_Y_INCHES),
                edgecolor="black",
                rect=[PLOT_LEFT, PLOT_BOT, PLOT_WIDTH, PLOT_HEIGHT])

        trajectory_fig.set_invert_x_axis() # Because UE4 uses left-handed 
                                           # coordinate system the X
                                           # axis in the graph is flipped
        trajectory_fig.set_axis_equal()    # X-Y spacing should be equal in size



        # Add waypoint markers
        trajectory_fig.add_graph("waypoints", window_size=len(waypoints),
                                 x0=waypoints[:,0], y0=waypoints[:,1],
                                 linestyle="-", marker="", color='g')
        # Add trajectory markers
        trajectory_fig.add_graph("trajectory", window_size=TOTAL_EPISODE_FRAMES,
                                 x0=[start_x]*TOTAL_EPISODE_FRAMES, 
                                 y0=[start_y]*TOTAL_EPISODE_FRAMES,
                                 color=[1, 0.5, 0])
        # Add starting position marker
        trajectory_fig.add_graph("start_pos", window_size=1, 
                                 x0=[start_x], y0=[start_y],
                                 marker=11, color=[1, 0.5, 0], 
                                 markertext="Start", marker_text_offset=1)

        trajectory_fig.add_graph("obstacles_points",
                                 window_size=8 * (NUM_PEDESTRIANS + NUM_VEHICLES) ,
                                 x0=[0]* (8 * (NUM_PEDESTRIANS + NUM_VEHICLES)), 
                                 y0=[0]* (8 * (NUM_PEDESTRIANS + NUM_VEHICLES)),
                                    linestyle="", marker="+", color='b')

        nearest_tl = []
        tl_dict = {}
        # we compute here traffic lights filter because they are stationary objects.
        for i,tl in enumerate(traffic_lights):
            # compute distances vector between waypoints and current traffic light
            temp = waypoints[:,:2] - tl[1:3]
            # compute module fpr each distances vector 
            dist = np.linalg.norm(temp,axis=1)
            # verify if there is at least one traffic_light 
            # along waypoints trajectory and plot it.
            # For each i-th waypoint we consider a circle of
            # radius 5 and centered in i-th waypoint. If traffic lights
            # point is in almost a circle we considered it.
            TRAFFIC_LIGHT_DISTANCE = 10 # sperimentaly computed
            if len(np.where(dist<TRAFFIC_LIGHT_DISTANCE)[0]>0):
                nearest_tl.append(tl[:-1]) # not interested to store status information  here
                #get id and status
                tl_dict[tl[0]]=tl[-1]
                if enable_live_plot:
                    trajectory_fig.add_graph(f"{tl[0]}",
                                        window_size=1, 
                                        x0=[tl[1]], y0=[tl[2]],
                                        marker=11, color=[1, 0.5, 0], 
                                        markertext=f"{i}", marker_text_offset=1)
            
        nearest_tl = np.array(nearest_tl)
        print("SHAPE:")
        print(nearest_tl.shape)
                
                    

        # Add end position marker
        trajectory_fig.add_graph("end_pos", window_size=1, 
                                 x0=[waypoints[-1, 0]], 
                                 y0=[waypoints[-1, 1]],
                                 marker="D", color='r', 
                                 markertext="End", marker_text_offset=1)
        # Add car marker
        trajectory_fig.add_graph("car", window_size=1, 
                                 marker="s", color='b', markertext="Car",
                                 marker_text_offset=1)
        # Add lead car information
        trajectory_fig.add_graph("leadcar", window_size=1, 
                                 marker="s", color='g', markertext="Lead Car",
                                 marker_text_offset=1)

        # Add lookahead path
        trajectory_fig.add_graph("selected_path", 
                                 window_size=INTERP_MAX_POINTS_PLOT,
                                 x0=[start_x]*INTERP_MAX_POINTS_PLOT, 
                                 y0=[start_y]*INTERP_MAX_POINTS_PLOT,
                                 color=[1, 0.5, 0.0],
                                 linewidth=3)

        # Add local path proposals
        for i in range(NUM_PATHS):
            trajectory_fig.add_graph("local_path " + str(i), window_size=200,
                                     x0=None, y0=None, color=[0.0, 0.0, 1.0])

        ###
        # Add 1D speed profile updater
        ###
        forward_speed_fig =\
                lp_1d.plot_new_dynamic_figure(title="Forward Speed (m/s)")
        forward_speed_fig.add_graph("forward_speed", 
                                    label="forward_speed", 
                                    window_size=TOTAL_EPISODE_FRAMES)
        forward_speed_fig.add_graph("reference_signal", 
                                    label="reference_Signal", 
                                    window_size=TOTAL_EPISODE_FRAMES)

        # Add throttle signals graph
        throttle_fig = lp_1d.plot_new_dynamic_figure(title="Throttle")
        throttle_fig.add_graph("throttle", 
                              label="throttle", 
                              window_size=TOTAL_EPISODE_FRAMES)
        # Add brake signals graph
        brake_fig = lp_1d.plot_new_dynamic_figure(title="Brake")
        brake_fig.add_graph("brake", 
                              label="brake", 
                              window_size=TOTAL_EPISODE_FRAMES)
        # Add steering signals graph
        steer_fig = lp_1d.plot_new_dynamic_figure(title="Steer")
        steer_fig.add_graph("steer", 
                              label="steer", 
                              window_size=TOTAL_EPISODE_FRAMES)

        # live plotter is disabled, hide windows
        if not enable_live_plot:
            lp_traj._root.withdraw()
            lp_1d._root.withdraw()        


        #############################################
        # Local Planner Variables
        #############################################
        wp_goal_index   = 0
        local_waypoints = None
        path_validity   = np.zeros((NUM_PATHS, 1), dtype=bool)
        lp = local_planner.LocalPlanner(NUM_PATHS,
                                        PATH_OFFSET,
                                        CIRCLE_OFFSETS,
                                        CIRCLE_RADII,
                                        PATH_SELECT_WEIGHT,
                                        TIME_GAP,
                                        A_MAX,
                                        SLOW_SPEED,
                                        STOP_LINE_BUFFER)
        bp = behavioural_planner.BehaviouralPlanner(BP_LOOKAHEAD_BASE,
                                                    LEAD_VEHICLE_LOOKAHEAD,
                                                    nearest_tl,
                                                    tl_dict)

        #############################################
        # Scenario Execution Loop
        #############################################

        # Iterate the frames until the end of the waypoints is reached or
        # the TOTAL_EPISODE_FRAMES is reached. The controller simulation then
        # ouptuts the results to the controller output directory.
        reached_the_end = False
        skip_first_frame = True

        # Initialize the current timestamp.
        current_timestamp = start_timestamp

        # Initialize collision history
        prev_collision_vehicles    = 0
        prev_collision_pedestrians = 0
        prev_collision_other       = 0


        # vehicles_dict = {}
        ####################################
        entering = {}
        outgoing = {}

        # the aboves data structure are structured in this way:
        # entering = {
        #          id1: [counter, agent_object],
        #          id2: [counter, agent_object],
        #          ....
        #          }
        #
    
        # list of last frame ids  
        last_frame_agents = []


    
        ###################################
        # DETECTOR

        net = load_model()

        for frame in range(TOTAL_EPISODE_FRAMES):
            # Gather current data from the CARLA server
            measurement_data, sensor_data = client.read_data()

            # UPDATE HERE the obstacles list
            obstacles = []

            _vehicles_dict = {}
           
            # Update pose and timestamp
            prev_timestamp = current_timestamp
            current_x, current_y, current_z, current_pitch, current_roll, current_yaw = \
                get_current_pose(measurement_data)
            current_speed = measurement_data.player_measurements.forward_speed
            current_timestamp = float(measurement_data.game_timestamp) / 1000.0

            # Wait for some initial time before starting the demo
            if current_timestamp <= WAIT_TIME_BEFORE_START:
                send_control_command(client, throttle=0.0, steer=0, brake=1.0)
                continue
            else:
                current_timestamp = current_timestamp - WAIT_TIME_BEFORE_START
            
            # Store history
            x_history.append(current_x)
            y_history.append(current_y)
            yaw_history.append(current_yaw)
            speed_history.append(current_speed)
            time_history.append(current_timestamp) 

            # Store collision history
            collided_flag,\
            prev_collision_vehicles,\
            prev_collision_pedestrians,\
            prev_collision_other = get_player_collided_flag(measurement_data,
                                                 prev_collision_vehicles,
                                                 prev_collision_pedestrians,
                                                 prev_collision_other)
                                                      
            collided_flag_history.append(collided_flag)

            if frame % (LP_FREQUENCY_DIVISOR) == 0:
            # update traffic_lights status
                ###################################
                # GET BGR

                camera_data = sensor_data.get('CameraRGB', None)
                if camera_data is not None:
                    # to_bgra_array returns an image with 4 channels with last channel all zeros
                    camera_data = to_bgra_array(camera_data)[:,:,:3]
                    camera_data = np.copy(camera_data)

                camera_data_bis = sensor_data.get("CameraRGBbis", None)
                if camera_data_bis is not None:
                    camera_data_bis = to_bgra_array(camera_data_bis)[:,:,:3]
                    camera_data_bis = np.copy(camera_data_bis)

                #output segmentation
                seg_data = sensor_data.get('CameraSemSeg', None)
                if seg_data is not None:
                    seg_data = seg_data.data

                seg_data_bis = sensor_data.get('CameraSemSegbis', None)
                if seg_data_bis is not None:
                    seg_data_bis = seg_data_bis.data

                #depth camera
                depth_data = sensor_data.get('CameraDepth', None)
                if depth_data is not None:
                    depth_data = depth_data.data

                depth_data_bis = sensor_data.get('CameraDepthbis', None)
                if depth_data_bis is not None:
                    depth_data_bis = depth_data_bis.data

                # print("-"*50)

                world_frame_vehicles, world_frame_pedestrians,sidewalk = find_pedestrians_and_vehicles_from_camera(net, camera_data, seg_data, depth_data, current_x, current_y, current_z, current_yaw, camera_parameters)
                wfv_bis, wfp_bis, sidewalk_bis = find_pedestrians_and_vehicles_from_camera(net, camera_data_bis, seg_data_bis, depth_data_bis, current_x, current_y, current_z,current_yaw, camera_parameters_bis, True)
                # world_frame_vehicles, world_frame_pedestrians,sidewalk = find_pedestrians_and_vehicles_from_camera(net, camera_data, seg_data, depth_data, current_x, current_y, current_yaw, camera_parameters)
                # wfv_bis, wfp_bis, sidewalk_bis = find_pedestrians_and_vehicles_from_camera(net, camera_data_bis, seg_data_bis, depth_data_bis, current_x, current_y, current_yaw, camera_parameters_bis, True)

                # world_frame_vehicles += wfv_bis
                # world_frame_pedestrians += wfp_bis

                # for p in world_frame_vehicles:
                #     print("CAMERA 0 vehicles ", p)

                # print()


                # for p in wfv_bis:
                #     print("CAMERA BIS vehicles ", p)

                # print()

                ###############################################
                
                # BELOW CARLA PERFECT DATA

                pedestrians = []
                vehicles = []
                for agent in measurement_data.non_player_agents:
                    if agent.HasField("traffic_light"):
                        if agent.id in tl_dict:
                            tl_dict[agent.id] = agent.traffic_light.state
                    if agent.HasField("pedestrian"):
                        location = agent.pedestrian.transform.location
                        dimensions = agent.pedestrian.bounding_box.extent
                        orientation = agent.pedestrian.transform.rotation
                        
                        dist = np.subtract([current_x,current_y], [location.x,location.y])
                        norm = np.linalg.norm(dist)
                        # filter only pedestrian that are in a radiud of 30 metres
                        
                        if norm < AGENTS_CHECK_RADIUS:
                            bb = obstacle_to_world(location, dimensions, orientation)
                            #takes only verteces of pedestrians bb
                            bb = bb[0:-1:2]
                            orientation = orientation.yaw*math.pi/180
                            speed = agent.pedestrian.forward_speed
                            # print("REAL PED: ", location.x,location.y)

                            pedestrian = Agent(agent.id,[location.x,location.y],bb,orientation,speed,"Pedestrian")
                            pedestrians.append(pedestrian)

                    if agent.HasField("vehicle"):
                        location = agent.vehicle.transform.location
                        dimensions = agent.vehicle.bounding_box.extent
                        orientation = agent.vehicle.transform.rotation

                        dist = np.subtract([current_x,current_y], [location.x,location.y])
                        norm = np.linalg.norm(dist)
                        # filter only vehicle that are in a radiud of AGENTS_CHECK_RADIUS metres
                        
                        if norm < AGENTS_CHECK_RADIUS:
                            id = agent.id
                            speed = agent.vehicle.forward_speed
                            bb = obstacle_to_world(location, dimensions, orientation)
                            #takes only verteces of pedestrians bb
                            bb = bb[0:-1:2]
                            # print("REAL VEHICLE: ", location.x,location.y)
                            vehicle = Agent(id,[location.x,location.y],bb,orientation.yaw,speed,"Vehicle")
                            vehicles.append(vehicle)
                            if id in outgoing:
                                # update its data because in the current frame this object can be still occludeed 
                                outgoing[id][1] = vehicle
                            if SIMULATION_PERFECT:
                                _vehicles_dict[id] = vehicle 
                            

                #########################################
                # here make data association (remember to valuate it only on x and y)
                # input-> world_frame_vehicles, world_frame_pedestrians, sidewalk
                # output-> np array di pedoni

                
                pedestrian_associated,_ = association_vehicle_pedestrian(pedestrians,
                world_frame_pedestrians,wfp_bis,sidewalk,sidewalk_bis,True)

                vehicles_associated, vehicles_dict = association_vehicle_pedestrian(vehicles,
                world_frame_vehicles,wfv_bis)


                # pedestrians_to_consider = pedestrian_associated
                # vehicles_to_consider = vehicles_associated

                pedestrians_to_consider = []
                vehicles_to_consider = []

                print("entering prima")
                
                for agent in entering.values():
                    print(str(agent[1]), "for", agent[0], "times")

                ########    entering  management 
                output_p, entering, _ = agent_entering_management(pedestrian_associated,last_frame_agents,entering)
                output_v, entering, vehicles_dict = agent_entering_management(vehicles_associated,last_frame_agents,entering,vehicles_dict)

                pedestrians_to_consider += output_p
                vehicles_to_consider += output_v

                print("AGENT_TO_CONSIDER dopo entering",len(pedestrians_to_consider), len(vehicles_to_consider))
                
                print("\nentering dopo")

                for agent in entering.values():
                    print(str(agent[1]), "for", agent[0], "times")

                ####### outgoing management
                print("outgoing prima")
                for agent in outgoing.values():
                    print(str(agent[1]), "for", agent[0], "times")

                output_p, outgoing, _ = agents_outgoing_managements(pedestrian_associated,last_frame_agents,outgoing)
                output_v, outgoing, vehicles_dict = agents_outgoing_managements(vehicles_associated,last_frame_agents,outgoing,vehicles_dict)

                pedestrians_to_consider += output_p
                vehicles_to_consider += output_v
                print("AGENT_TO_CONSIDER dopo outgoing",len(pedestrians_to_consider), len(vehicles_to_consider))
                
                print("outgoing dopo")
                for agent in outgoing.values():
                    print(str(agent[1]), "for", agent[0], "times")

                last_frame_agents = pedestrians_to_consider+vehicles_to_consider               
                #######

                if SIMULATION_PERFECT:
                    vehicles_dict = _vehicles_dict


                if not SIMULATION_PERFECT:
                    pedestrians = np.array(pedestrians_to_consider)
                    vehicles = np.array(vehicles_to_consider)
                else:
                    pedestrians = np.array(pedestrians,dtype=object)
                    vehicles = np.array(vehicles)

                # set current info about traffic light (status), pedestrian and vehicle 
                bp.set_tl_dict(tl_dict)
                bp.set_pedestrians(pedestrians)
                bp.set_vehicles(vehicles)
                bp.set_vehicles_dict(vehicles_dict)

            camera_data = sensor_data.get('CameraRGBView', None)
            if camera_data is not None:
                camera_data = to_bgra_array(camera_data)[:,:,:3]
                cv2.imshow("CameraRGB", camera_data)
                cv2.waitKey(10)




                
            # Execute the behaviour and local planning in the current instance
            # Note that updating the local path during every controller update
            # produces issues with the tracking performance (imagine everytime
            # the controller tried to follow the path, a new path appears). For
            # this reason, the local planner (LP) will update every X frame,
            # stored in the variable LP_FREQUENCY_DIVISOR, as it is analogous
            # to be operating at a frequency that is a division to the 
            # simulation frequency.
            if frame % LP_FREQUENCY_DIVISOR == 0:
                # Compute open loop speed estimate.
                open_loop_speed = lp._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp)

                # Calculate the goal state set in the local frame for the local planner.
                # Current speed should be open loop for the velocity profile generation.
                ego_state = [current_x, current_y, current_yaw, open_loop_speed]

                # Set lookahead based on current speed.
                bp.set_lookahead(BP_LOOKAHEAD_BASE + BP_LOOKAHEAD_TIME * open_loop_speed)


                if False:
                    if WINDOWS_OS:
                        os.system("cls")
                    else:
                        os.system("clear")

                    print(f"[LOGINFO]: from {args.start} to {args.dest}\t[DESIRED_SPEED]: {DESIRED_SPEED} m/s")
                    print(f"[PEDESTRIANS]: {NUM_PEDESTRIANS}, {SEED_PEDESTRIANS}\t[VEHICLES]: {NUM_VEHICLES}, {SEED_VEHICLES}\n")

                    # Perform a state transition in the behavioural planner.
                    bp.transition_state(waypoints, ego_state, current_speed)

                    states = ["FOLLOW_LANE", "DECELERATE_TO_STOP", "STAY_STOPPED"]
                    print(f"[CURRENT_STATE]: {states[bp._state]}", end="\t")
                    print(f"[COLLISION]: {'Yes' if collided_flag else 'No'}")

                    print(f"[EGO_POS]: ({round(current_x, 2)}, {round(current_y, 2)})", end='\t')
                    print(f"[EGO_YAW]: {round(current_yaw*180/math.pi, 2)} deg", end='\t')
                    print(f"[EGO_SPEED]: {round(current_speed,2)} m/s")

                    print(f"[PEDESTRIAN_COLLISION_PREDICTED]: {'Yes' if bp._pedestrian_detected else 'No'}")
                    print(f"[VEHICLE_COLLISION_PREDICTED]: {'Yes' if bp._car_collision_predicted else 'No'}")

                    # print(f"[PED_POS]: (XXX.XX, XXX.XX)", end='\t')
                    # print(f"[PED_YAW]: X.XX deg", end='\t')
                    # print(f"[PED_SPEED]: X.XX m/s")

                    leader = bp._lead_vehicle
                    if leader is None:
                        print(f"[LEAD_POS]: (XXX.XX, XXX.XX)", end='\t')
                        print(f"[LEAD_YAW]: X.XX deg", end='\t')
                        print(f"[LEAD_SPEED]: X.XX m/s")
                    else:
                        leader_pos = leader.get_position()
                        print(f"[LEAD_POS]: ({round(leader_pos[0], 2)}, {round(leader_pos[1], 2)})", end='\t')
                        print(f"[LEAD_YAW]: {round(leader.get_orientation(), 2)} deg", end='\t')
                        print(f"[LEAD_SPEED]: {round(leader.get_speed(), 2)} m/s")

                    tl = bp._current_traffic_light
                    if len(tl) != 0:
                        print(f"[T_LIG_POS]: ({round(tl[1],2)}, {round(tl[2],2)})", end='\t')
                        print(f"[T_LIG_YAW]: {round(tl[3],2)} deg", end='\t')
                        statuses = ["GREEN", "YELLOW", "RED"]
                        print(f"[T_LIG_STATUS]: {statuses[bp._tl_dict[tl[0]]]}")
                    else:
                        print(f"[T_LIG_POS]: (XXX.XX, XXX.XX)", end='\t')
                        print(f"[T_LIG_YAW]: X.XX deg", end='\t')
                        print(f"[T_LIG_STATUS]: X.XX m/s")
                else:
                    bp.transition_state(waypoints, ego_state, current_speed)

                # Compute the goal state set from the behavioural planner's computed goal state.
                goal_state_set = lp.get_goal_state_set(bp._goal_index, bp._goal_state, waypoints, ego_state)

                # Calculate planned paths in the local frame.
                paths, path_validity = lp.plan_paths(goal_state_set)

                # Transform those paths back to the global frame.
                paths = local_planner.transform_paths(paths, ego_state)

                # Perform collision checking.
                collision_check_array = lp._collision_checker.collision_check(paths, [])

                # Compute the best local path.
                best_index = lp._collision_checker.select_best_path_index(paths, collision_check_array, bp._goal_state)
                # If no path was feasible, continue to follow the previous best path.
                if best_index == None:
                    best_path = lp._prev_best_path
                else:
                    best_path = paths[best_index]
                    lp._prev_best_path = best_path

                if best_path is not None:
                    # Compute the velocity profile for the path, and compute the waypoints.
                    desired_speed = bp._goal_state[2]
                    decelerate_to_stop = bp._state == behavioural_planner.DECELERATE_TO_STOP
                    
                    lead_car_state = None
                    if bp._lead_vehicle is not None:
                        lead_car_pos = bp._lead_vehicle.get_position()
                        lead_car_speed = bp._lead_vehicle.get_speed()
                        lead_car_state = [lead_car_pos[0],lead_car_pos[1],lead_car_speed] 
                    local_waypoints = lp._velocity_planner.compute_velocity_profile(best_path, desired_speed, ego_state, current_speed, decelerate_to_stop, lead_car_state, bp._follow_lead_vehicle)

                    if local_waypoints != None:
                        # Update the controller waypoint path with the best local path.
                        # This controller is similar to that developed in Course 1 of this
                        # specialization.  Linear interpolation computation on the waypoints
                        # is also used to ensure a fine resolution between points.
                        wp_distance = []   # distance array
                        local_waypoints_np = np.array(local_waypoints)
                        for i in range(1, local_waypoints_np.shape[0]):
                            wp_distance.append(
                                    np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i-1, 0])**2 +
                                            (local_waypoints_np[i, 1] - local_waypoints_np[i-1, 1])**2))
                        wp_distance.append(0)  # last distance is 0 because it is the distance
                                            # from the last waypoint to the last waypoint

                        # Linearly interpolate between waypoints and store in a list
                        wp_interp      = []    # interpolated values 
                                            # (rows = waypoints, columns = [x, y, v])
                        for i in range(local_waypoints_np.shape[0] - 1):
                            # Add original waypoint to interpolated waypoints list (and append
                            # it to the hash table)
                            wp_interp.append(list(local_waypoints_np[i]))
                    
                            # Interpolate to the next waypoint. First compute the number of
                            # points to interpolate based on the desired resolution and
                            # incrementally add interpolated points until the next waypoint
                            # is about to be reached.
                            num_pts_to_interp = int(np.floor(wp_distance[i] /\
                                                        float(INTERP_DISTANCE_RES)) - 1)
                            wp_vector = local_waypoints_np[i+1] - local_waypoints_np[i]
                            wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

                            for j in range(num_pts_to_interp):
                                next_wp_vector = INTERP_DISTANCE_RES * float(j+1) * wp_uvector
                                wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))
                        # add last waypoint at the end
                        wp_interp.append(list(local_waypoints_np[-1]))
                        
                        # Update the other controller values and controls
                        controller.update_waypoints(wp_interp)

            ###
            # Controller Update
            ###
            if local_waypoints != None and local_waypoints != []:
                controller.update_values(current_x, current_y, current_yaw, 
                                         current_speed,
                                         current_timestamp, frame)
                controller.update_controls()
                cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()
            else:
                cmd_throttle = 0.0
                cmd_steer = 0.0
                cmd_brake = 0.0

            # Skip the first frame or if there exists no local paths
            if skip_first_frame and frame == 0:
                pass
            elif local_waypoints == None:
                pass
            else:
                # Update live plotter with new feedback
                trajectory_fig.roll("trajectory", current_x, current_y)
                trajectory_fig.roll("car", current_x, current_y)

                if lead_car_state is not None:
                    current_lead_car_x = lead_car_state[0]
                    current_lead_car_y = lead_car_state[1]
                else:
                    current_lead_car_x = 0
                    current_lead_car_y = 0

                trajectory_fig.roll("leadcar", current_lead_car_x, current_lead_car_y)


                
                # Load parked car points
                obstacles = np.array(obstacles)
                if len(obstacles) > 0:
                    x = obstacles[:,:,0]
                    y = obstacles[:,:,1]

                    trajectory_fig.roll("obstacles_points", x, y)

                
                forward_speed_fig.roll("forward_speed", 
                                       current_timestamp, 
                                       current_speed)
                forward_speed_fig.roll("reference_signal", 
                                       current_timestamp, 
                                       controller._desired_speed)
                throttle_fig.roll("throttle", current_timestamp, cmd_throttle)
                brake_fig.roll("brake", current_timestamp, cmd_brake)
                steer_fig.roll("steer", current_timestamp, cmd_steer)

                # Local path plotter update
                if frame % LP_FREQUENCY_DIVISOR == 0:
                    path_counter = 0
                    for i in range(NUM_PATHS):
                        # If a path was invalid in the set, there is no path to plot.
                        if path_validity[i]:
                            # Colour paths according to collision checking.
                            if not collision_check_array[path_counter]:
                                colour = 'r'
                            elif i == best_index:
                                colour = 'k'
                            else:
                                colour = 'b'
                            trajectory_fig.update("local_path " + str(i), paths[path_counter][0], paths[path_counter][1], colour)
                            path_counter += 1
                        else:
                            trajectory_fig.update("local_path " + str(i), [ego_state[0]], [ego_state[1]], 'r')
                # When plotting lookahead path, only plot a number of points
                # (INTERP_MAX_POINTS_PLOT amount of points). This is meant
                # to decrease load when live plotting
                wp_interp_np = np.array(wp_interp)
                path_indices = np.floor(np.linspace(0, 
                                                    wp_interp_np.shape[0]-1,
                                                    INTERP_MAX_POINTS_PLOT))
                trajectory_fig.update("selected_path", 
                        wp_interp_np[path_indices.astype(int), 0],
                        wp_interp_np[path_indices.astype(int), 1],
                        new_colour=[1, 0.5, 0.0])


                # Refresh the live plot based on the refresh rate 
                # set by the options
                if enable_live_plot and \
                   live_plot_timer.has_exceeded_lap_period():
                    lp_traj.refresh()
                    lp_1d.refresh()
                    live_plot_timer.lap()
            

            # Output controller command to CARLA server
            send_control_command(client,
                                 throttle=cmd_throttle,
                                 steer=cmd_steer,
                                 brake=cmd_brake)

            # Find if reached the end of waypoint. If the car is within
            # DIST_THRESHOLD_TO_LAST_WAYPOINT to the last waypoint,
            # the simulation will end.
            dist_to_last_waypoint = np.linalg.norm(np.array([
                waypoints[-1][0] - current_x,
                waypoints[-1][1] - current_y]))
            
            
            if  dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT:
                reached_the_end = True
            if reached_the_end:
                break

        # End of demo - Stop vehicle and Store outputs to the controller output
        # directory.
        if reached_the_end:
            print("Reached the end of path. Writing to controller_output...")
        else:
            print("Exceeded assessment time. Writing to controller_output...")
        # Stop the car
        send_control_command(client, throttle=0.0, steer=0.0, brake=1.0)
        # Store the various outputs
        store_trajectory_plot(trajectory_fig.fig, 'trajectory.png')
        store_trajectory_plot(forward_speed_fig.fig, 'forward_speed.png')
        store_trajectory_plot(throttle_fig.fig, 'throttle_output.png')
        store_trajectory_plot(brake_fig.fig, 'brake_output.png')
        store_trajectory_plot(steer_fig.fig, 'steer_output.png')
        write_trajectory_file(x_history, y_history, speed_history, time_history,
                              collided_flag_history)
        write_collisioncount_file(collided_flag_history)
    
def main():
    """Main function.

    Args:
        -v, --verbose: print debug information
        -l, --local: use local server
        -s, --start: player start index
        -d, --dest: player destination index
        -a, --autopilot: enable autopilot
        -q, --quality-level: graphics quality level [Low or Epic]
        -i, --images-to-disk: save images to disk
        -c, --carla-settings: Path to CarlaSettings.ini file
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--local', '-l',
        action='store_true',
        dest = 'local'
    )
    argparser.add_argument(
        '-s', '--start',
        metavar='S',
        default = PLAYER_START_INDEX,
        type=int,
        help='Player start index')
    argparser.add_argument(
        '-d', '--dest',
        metavar='D',
        default = DESTINATION_INDEX,
        type=int,
        help='Player destination index')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level.')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    args = argparser.parse_args()

    # Logging startup info
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    if not args.local:
        host = SERVER_HOST; port = SERVER_PORT
    else:
        host = LOCAL_HOST; port = LOCAL_PORT
        #host = "192.168.1.128"; port = 2000
    
    logging.info('listening to server %s:%s', host, port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    # Execute when server connection is established
    while True:
        try:
            exec_waypoint_nav_demo(args, host, port)
            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

