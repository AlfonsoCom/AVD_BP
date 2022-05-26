#!/usr/bin/env python3

from __future__ import print_function

import argparse
import logging
import random
import time
import cv2
import numpy as np
import pygame
import math
from math import cos, sin, pi,tan

import controller2d_AR as controller2d

#   Required to import carla library
import os
import sys
sys.path.append(os.path.abspath(sys.path[0] + '/..'))

# Imports detector
detector_path = 'D:/Dati/AutonomousDriving/carla-car-detection/'
sys.path.append(os.path.abspath(detector_path))
sys.path.append(detector_path+"models/research/object_detection/")
sys.path.append(detector_path+"models/research/")

from detection import detector

from camera_geometry import CameraGeometry
from lane_detection import lane_detection, image_filter_and_crop

import curved_lane_detection
from matplotlib import pyplot as plt

# Carla Imports
from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
import carla.image_converter as image_converter
from carla.planner.city_track import CityTrack
from carla.planner.map import CarlaMap


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

# Camera parameters
camera_parameters = {}
camera_parameters['x'] = 1.8
camera_parameters['y'] = 0
camera_parameters['z'] = 1.3
camera_parameters['width'] = 800
camera_parameters['height'] = 600
camera_parameters['fov'] = 90


NUM_PEDESTRIANS        = 0      # total number of pedestrians to spawn
NUM_VEHICLES           = 2      # total number of vehicles to spawn
SEED_PEDESTRIANS       = 0      # seed for pedestrian spawn randomizer
SEED_VEHICLES          = 0     # seed for vehicle spawn randomizer
CLIENT_WAIT_TIME       = 3      # wait time for client before starting episode
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

    # Adding sensors: 

    # Common cameras settings
    cam_height = camera_parameters['z']
    cam_x_pos = camera_parameters['x']
    cam_y_pos = camera_parameters['y']
    camera_width = camera_parameters['width']
    camera_height = camera_parameters['height']
    camera_fov = camera_parameters['fov']

    # RGB Camera
    camera0 = Camera("CameraRGB")
    camera0.set_image_size(camera_width, camera_height)
    camera0.set(FOV=camera_fov)
    camera0.set_position(cam_x_pos, cam_y_pos, cam_height)

    settings.add_sensor(camera0)


    # Segmentation Camera
    camera1 = Camera("Segmentation", PostProcessing="SemanticSegmentation")

    camera1.set_image_size(camera_width, camera_height)
    camera1.set(FOV=camera_fov)
    camera1.set_position(cam_x_pos, cam_y_pos, cam_height)

    settings.add_sensor(camera1)
    
    # Depth Camera
    camera2 = Camera("Depth", PostProcessing="Depth")

    camera2.set_image_size(camera_width, camera_height)
    camera2.set(FOV=camera_fov)
    camera2.set_position(cam_x_pos, cam_y_pos, cam_height)

    settings.add_sensor(camera2)
    

    return settings


def run_carla_client(args):
    # Here we will run 3 episodes with 300 frames each.
    number_of_episodes = 1
    frames_per_episode = 10000
    synchronous_mode = True

    # Create the connection with the server already connected at args.host : args.port
    with make_carla_client(args.host, args.port, timeout = None) as client:
        print('CarlaClient connected')

        for episode in range(0, number_of_episodes):
            # Start a new episode : 
            #   Each episode has an own setup
            #   A single connection can manage more  episodes
            settings = make_carla_settings(args)
            
            # Loading settings into the server.
            # Scene object with map_name, startins_spots and sensors configured.
            scene = client.load_settings(settings)
            
            # Visualize the possible starting position and choose one
            # print("Starting Position : ",scene.player_start_spots)
            player_start = random.randint(0, max(0, len(scene.player_start_spots) - 1))

            # Starting the episode at player_start index
            print('Starting new episode at %r' % scene.map_name)
            print('Sensors loaded %r' % scene.sensors)
            print(player_start)
            
            controller = None
            autopilot_enable = False
            LP_divider = 1

            client.start_episode(player_start)

            vehicle_detector = detector(detector_path + "full_trained_model/detectors-40264/")

            camera_width = camera_parameters['width']
            camera_height = camera_parameters['height']
            camera_fov = camera_parameters['fov']
            cam_z_pos = camera_parameters['z']
            cam_x_pos = camera_parameters['x']
            cam_y_pos = camera_parameters['y']


            f = camera_width * 0.5 /(2 * tan(camera_fov * pi / 360))
            Center_X = camera_width / 2.0
            Center_Y = camera_height / 2.0

            inv_intrinsic_matrix = np.linalg.inv( np.array([[f, 0, Center_X],[0, f, Center_Y], [0, 0, 1]]) )

            for i in range(frames_per_episode):
                measurements, sensors_data = client.read_data()

                cam_data = sensors_data.get("CameraRGB", None)

                segmentation_data = sensors_data.get("Segmentation", None)

                depth_data = sensors_data.get("Depth", None)

                
                x   = measurements.player_measurements.transform.location.x + cam_x_pos
                y   = measurements.player_measurements.transform.location.y + cam_y_pos
                z   = cam_z_pos
                yaw = math.radians(measurements.player_measurements.transform.rotation.yaw)

                A = np.zeros((4,4))

                R = np.dot(rotate_z(yaw + 90 * pi / 180),rotate_x(90 * pi / 180))

                A[:3,:3] = R
                A[:,-1]  = [x, y, z, 1]

                for agent in measurements.non_player_agents:
                    if agent.HasField('vehicle'):
                        # print(agent.vehicle.bounding_box.transform)
                        # print(agent.vehicle.bounding_box.extent)
                        # print(agent.vehicle.transform)
                        pass
                
                if cam_data is not None and depth_data is not None :
                    cam_data = image_converter.to_bgra_array(cam_data)
                    cam_data = cam_data[:,:,:3]
                    height, width, _ = cam_data.shape
                    frame, boxes = vehicle_detector.draw_boxes_for_image(cam_data , 0.3)

                    depth_data = image_converter.depth_to_array(depth_data) * 1000
                    world_frame_list = []

                    for box in boxes:
                        x1,x2,y1,y2 = box
                        x1 = int(x1 * (width - 1))
                        x2 = int(x2 * (width - 1))
                        y1 = int(y1 * (height - 1))
                        y2 = int(y2 * (height - 1))

                        cam_data = np.array(cam_data, np.uint8)
                        
                        cv2.rectangle(cam_data, (x1,y1), (x2,y2), (255,0,0), 2)

                        pixel = [x1, y1, 1]
                        camera_frame_point = np.dot(inv_intrinsic_matrix, pixel) * depth_data[y1][x1]
                        world_frame_point = np.zeros((4,1))
                        camera_frame_point = np.reshape(camera_frame_point,(3,1))
                        world_frame_point[:3] = camera_frame_point
                        world_frame_point[-1] = 1
                        world_frame_point = np.dot(A, world_frame_point)
                        world_frame_point = world_frame_point[:3]

                        world_frame_list.append(world_frame_point)


                    cv2.imshow("FRAME ", cam_data)
                    cv2.waitKey(10)
                    # print("Boxes : ", boxes)
                    print("Current Position : ", measurements.player_measurements.transform.location.x, measurements.player_measurements.transform.location.y)
                    
                    print("World Frame List : ", world_frame_point)

                    

                client.send_control(measurements.player_measurements.autopilot_control)
                

                

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-ww', '--window_width',
        default=200,
        type=int,
        help='window width')
    argparser.add_argument(
        '-wh', '--window_height',
        default=200,
        type=int,
        help='window height')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    while True:
        try:
            run_carla_client(args)

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

