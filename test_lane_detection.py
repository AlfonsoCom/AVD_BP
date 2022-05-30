#!/usr/bin/env python3

from __future__ import print_function

import argparse
import logging
import random
import time
import cv2
import numpy as np
import pygame
from math import cos, sin, pi,tan

import controller2d_AR as controller2d

#   Required to import carla library
import os
import sys
sys.path.append(os.path.abspath(sys.path[0] + '/..'))

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

def to_rot(r):
    Rx = np.mat([[ 1,         0,           0],
                 [ 0, cos(r[0]), -sin(r[0]) ],
                 [ 0, sin(r[0]),  cos(r[0]) ]])

    Ry = np.mat([[ cos(r[1]), 0,  sin(r[1]) ],
                 [ 0,         1,          0 ],
                 [-sin(r[1]), 0,  cos(r[1]) ]])

    Rz = np.mat([[ cos(r[2]), -sin(r[2]), 0 ],
                 [ sin(r[2]),  cos(r[2]), 0 ],
                 [         0,          0, 1 ]])

    return Rz*Ry*Rx

class LaneFollowing:
    def __init__(self,city):
        self.carla_map = CityTrack(city)
        self.map = CarlaMap(city, self.carla_map.get_pixel_density(), self.carla_map.get_node_density())

        self.cam_height = 1.3
        cam_x_pos = 1.80
        cam_y_pos = 0
        cam_pitch = 0

        cam_yaw = 90
        cam_roll = 0 #-90
        camera_width = 800

        camera_height = 600
        camera_fov = 90

        cam_rotation = np.array([cam_pitch * pi /180,cam_yaw * pi /180 ,cam_roll * pi /180])

        f = camera_width * 0.5 /(2 * tan(camera_fov * pi / 360))
        Center_X = camera_width / 2.0
        Center_Y = camera_height / 2.0

        self.inv_intrinsic_matrix = np.linalg.inv( np.array([[f, 0, Center_X],[0, f, Center_Y], [0, 0, 1]]) )

        rotation_image_camera_frame = np.dot(rotate_z(-90 * pi /180),rotate_x(-90 * pi /180))

        image_camera_frame = np.zeros((4,4))
        image_camera_frame[:3,:3] = rotation_image_camera_frame
        # image_camera_frame[:, -1] = [cam_x_pos, cam_y_pos, self.cam_height, 1]
        image_camera_frame[:, -1] = [0, 0, 0, 1]



        self.cam_to_vehicle_frame = lambda object_camera_frame: np.dot(image_camera_frame , object_camera_frame)
        

    def detect(self, image, transform, image_rgb, depth_data, advanced_lane = True, show_lines = False):
        height, width = image.shape
        vehicle_frame_list = []
        
        if advanced_lane:
            # Filter image 
            dst = image_filter_and_crop(image)

            if show_lines:
                cv2.imshow("Image_original", image_rgb)
                cv2.waitKey(1)
                cv2.imshow('Filter', dst)
                cv2.waitKey(1)

            # Perspective change
            dst = curved_lane_detection.perspective_warp(dst, dst_size=(image.shape[1],image.shape[0]))

            if show_lines:
                cv2.imshow('Filter+Perspective Tform', dst)
                cv2.waitKey(1)
            
            # Sliding Windows detection
            try:
                out_img, curves, lanes, ploty = curved_lane_detection.sliding_window(dst)
            except:
                return []

            y_value =  np.linspace(height//2 , height , height//2)
            lanes_avg = np.average(lanes, axis=0)

            x_value = lanes_avg[0]*y_value**2 + lanes_avg[1]*y_value + lanes_avg[2]

            # Check Bounds
            x_value[np.where(x_value < 0)] = 0
            x_value[np.where(x_value > width - 1)] = width - 1
            y_value[np.where(y_value < 0)] = 0
            y_value[np.where(y_value > height - 1)] = height - 1

            centerline = list(zip(x_value,y_value))
            centerline = np.asarray(centerline, np.int32)

            pts_cent = centerline.reshape((-1, 1, 2))

            curve_image = np.zeros_like(image)

            curve_image = cv2.polylines(curve_image, pts_cent , True, 255, 3)

            curve_image = curved_lane_detection.inv_perspective_warp(curve_image, dst_size=(image.shape[1],image.shape[0]))
            
            if show_lines:
                cv2.imshow('Original_Frame_CenterLine', curve_image)
                cv2.waitKey(1)

            img_ = curved_lane_detection.draw_lanes(image_rgb, curves[0], curves[1],image_shape=(image.shape[1],image.shape[0]))

            curve_image = np.dstack([curve_image,curve_image,curve_image,curve_image])
            img_ = cv2.addWeighted(img_, 1.0, curve_image, 1.0, 1.0)

            if show_lines:
                cv2.imshow('Sliding window+Curve Fit', out_img)
                cv2.waitKey(1)
                cv2.imshow('Overlay Lanes', img_)
                cv2.waitKey(1)

            points = reversed(centerline)

        else:
            lane_image, lanes = lane_detection(image, show_intermediate_steps=show_lines)
        
            if lanes is None or len(lanes) == 0:
                print("No waypoints!")
                return []
            elif len(lanes) == 2:
                # Lanes are 2 ==> central lane is avg
                centralLine = (
                    (lanes[0][0] + lanes[1][0]) // 2,
                    (lanes[0][1] + lanes[1][1]) // 2,
                    (lanes[0][2] + lanes[1][2]) // 2,
                    (lanes[0][3] + lanes[1][3]) // 2
                )
            elif len(lanes) == 1:
                # Lanes are 2 ==> central lane is avg
                x1,y1,x2,y2 = lanes[0]

                if y1 < y2:
                    lowest_x = x2
                    goal_point = (x1,y1)
                else:
                    lowest_x = x1
                    goal_point = (x2,y2)
                
                if lowest_x < width // 2:
                    # Left
                    direction = -100
                else:
                    # Right 
                    direction = 100
                
                centralLine = (
                    goal_point[0] + direction, 
                    goal_point[1] + abs(direction),
                    goal_point[0], 
                    goal_point[1],
                )


            slope, intercept = np.polyfit((centralLine[0],centralLine[2]),(centralLine[1],centralLine[3]),1)

            if show_lines:
                cv2.line(lane_image, (centralLine[0],centralLine[1]),(centralLine[2],centralLine[3]), (255, 0, 0), 10)

                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGRA2BGR)
                image_rgb = np.array(image_rgb, np.uint8)
                
                lane_image = cv2.addWeighted(image_rgb, 1.0, lane_image, 2.0, 0.0)

                cv2.imshow("Lane", lane_image)
                cv2.waitKey(10)

            x_line = lambda y :min(max(int(float(y - intercept) / slope), 0), height - 1)

            # min_y = min(centralLine[1], centralLine[3])
            # max_y = max(centralLine[1], centralLine[3])

            y = list(range(height - 1, height // 2, -1))

            x = map(x_line, y)

            points = list(zip(x,y))

        for x,y in points:
            # print("Point(x,y) : ",x,y)

            pixel = [x , y, 1]
            pixel = np.reshape(pixel, (3,1))
            

            # Projection Image to Camera Frame
            normal = np.reshape((0,0,1), (3,1))
            # rotation = [[],[],[]]
            # normal_c = np.dot(self.rotation_cam_to_road.T , normal)
            # lm = self.cam_height / ( np.dot( normal_c.T ,np.dot(self.inv_intrinsic_matrix,pixel)))
            a = np.dot(self.inv_intrinsic_matrix, pixel) 
            depth = depth_data[y][x] * 1000
            cam_frame_vect = a * depth
            
            cam_frame_vect_extended = np.zeros((4,1))
            cam_frame_vect_extended[:3] = cam_frame_vect 
            cam_frame_vect_extended[-1] = 1
            
            # Projection Camera to Vehicle Frame
            vehicle_frame = self.cam_to_vehicle_frame(cam_frame_vect_extended)

            vehicle_frame = vehicle_frame[:3]
            vehicle_frame = np.asarray(np.reshape(vehicle_frame, (1,3)))

            # 0.05 * (len(vehicle_frame_list) + 1)
            # vehicle_frame[0][0]

            vehicle_frame_list.append([vehicle_frame[0][0],-vehicle_frame[0][1] , 10.0])
        
        return vehicle_frame_list  

def make_settings(args,sensor_select,synchronous_mode):
    if args.settings_filepath is None:
        # Create a CarlaSettings object.
        settings = CarlaSettings()
        settings.set(
            SynchronousMode=synchronous_mode,
            SendNonPlayerAgentsInfo=False,
            NumberOfVehicles=10,
            NumberOfPedestrians=10,
            WeatherId=random.choice([1, 3, 7, 8, 14]),
            QualityLevel=args.quality_level)
        settings.randomize_seeds()

        # The default camera captures RGB images of the scene.
        # Sensor name identifies the instance.
        camera0 = Camera("CameraRGB")

        # set pixel Resolution: WIDTH * HEIGHT
        cam_height = 1.3
        cam_x_pos = 1.80
        cam_y_pos = 0
        cam_pitch = 0
        cam_yaw = 0
        cam_roll = 0
        camera_width = 800
        camera_height = 600
        camera_fov = 90

        camera0.set_image_size(camera_width, camera_height)
        camera0.set(FOV=camera_fov)

        # set position X (front), Y (lateral), Z (height) relative to the car in meters
        # (0,0,0) is at center of baseline of car 
        camera0.set_position(cam_x_pos, cam_y_pos, cam_height)
        camera0.set_rotation(cam_pitch, cam_yaw, cam_roll)

        settings.add_sensor(camera0)


        camera1 = Camera("Segmentation", PostProcessing="SemanticSegmentation")

        camera1.set_image_size(camera_width, camera_height)
        camera1.set(FOV=camera_fov)

        camera1.set_position(cam_x_pos, cam_y_pos, cam_height)
        camera1.set_rotation(cam_pitch, cam_yaw, cam_roll)
        settings.add_sensor(camera1)
        

        camera2 = Camera("Depth", PostProcessing="Depth")

        camera2.set_image_size(camera_width, camera_height)
        camera2.set(FOV=camera_fov)

        camera2.set_position(cam_x_pos, cam_y_pos, cam_height)
        camera2.set_rotation(cam_pitch, cam_yaw, cam_roll)
        settings.add_sensor(camera2)
        

    else:
        # Load the ClientSettings.ini
        with open(args.settings_filepath, 'r') as fp:
            settings = fp.read()
    
    return settings


def run_carla_client(args):
    # Here we will run 3 episodes with 300 frames each.
    number_of_episodes = 1
    frames_per_episode = 10000
    synchronous_mode = True


    lanes = LaneFollowing("Town01")

    # Create the connection with the server already connected at args.host : args.port
    with make_carla_client(args.host, args.port, timeout = None) as client:
        print('CarlaClient connected')

        for episode in range(0, number_of_episodes):
            # Start a new episode : 
            #   Each episode has an own setup
            #   A single connection can manage more  episodes
            settings = make_settings(args,episode,synchronous_mode)
            
            # Loading settings into the server.
            # Scene object with map_name, startins_spots and sensors configured.
            scene = client.load_settings(settings)
            
            # Visualize the possible starting position and choose one
            # print("Starting Position : ",scene.player_start_spots)
            player_start = random.randint(0, max(0, len(scene.player_start_spots) - 1))

            # Starting the episode at player_start index
            print('Starting new episode at %r' % scene.map_name)
            print('Sensors loaded %r' % scene.sensors)
            
            controller = None
            autopilot_enable = False
            LP_divider = 1

            client.start_episode(player_start)

            for i in range(frames_per_episode):
                measurements, sensors_data = client.read_data()

                cam_data = sensors_data.get("CameraRGB", None)

                segmentation_data = sensors_data.get("Segmentation", None)

                depth_data = sensors_data.get("Depth", None)

                transform = measurements.player_measurements.transform

                if controller is None:
                    controller = controller2d.Controller2D([[transform.location.x, transform.location.y, 0.0],[transform.location.x, transform.location.y, 0.0]])

                if cam_data is not None and segmentation_data is not None and depth_data is not None:
                    if measurements.frame_number % LP_divider == 0:
                        segmentation_data = image_converter.labels_to_array(segmentation_data) 
                        depth_data = image_converter.depth_to_array(depth_data)
                        vehicle_frame_list = lanes.detect(segmentation_data, transform, image_converter.to_bgra_array(cam_data), depth_data, show_lines = True)

                        
                        # If exists waypoints, updates controller
                        if vehicle_frame_list is not None and len(vehicle_frame_list) > 1:
                            controller.update_waypoints(vehicle_frame_list)                         
                            # - ==> SINISTRA (WAY == STERZO)
                        


                    # Local Update of controller
                    controller.update_values(
                        0.0,
                        0.0,
                        0.0, 
                        measurements.player_measurements.forward_speed,
                        float(measurements.game_timestamp) / 1000.0,
                        measurements.frame_number
                        )

                    controller.update_controls()
                    throttle, steer, brake = controller.get_commands()

                    # print("Throttle : {0} \n Steer : {1} \n Brake : {2} \n ".format(throttle, steer, brake ))

                    client.send_control(throttle=throttle, steer=steer, brake=brake)
                

                

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

