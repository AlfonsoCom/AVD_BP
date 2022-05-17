import numpy as np
import math
from utils import *
from shapely.geometry import Point, Polygon

def detect_lead_vehicle(ego_pos,ego_yaw,vehicles,lookahead,looksideways_right=2.5,looksideways_left=1.5):
    if len(vehicles)==0:
        return None

    # Step 1 filter vehicles in bb
    A,B,C,D = compute_bb_verteces(ego_pos,lookahead,ego_yaw,looksideways_right,looksideways_left)
    bb = Polygon([A,B,C,D,A])

    vehicles_boolean = vehicles == None
    flag = False
    for i,vehicle in enumerate(vehicles):
        vehicle_bb_verteces = vehicle.get_bounding_box()

        for bb_vertex in vehicle_bb_verteces:
            vertex = Point(bb_vertex)
            if bb.contains(vertex):
                flag = True
                vehicles_boolean[i] = True
                break
    
    vehicles = vehicles[vehicles_boolean]

    if len(vehicles) == 0:
        return None

    # STEP 2 check if car orientation is equal to the orientation of filtered cars
    # because a lead vehicle moves itself in the same  direction of the ego car 

    THRESHOLD_DEGREE = 3.5
    
    ego_yaw = ego_yaw*180/math.pi
    vehicles_boolean = vehicles == None

    for i, vehicle in enumerate(vehicles):
        check_sum = abs(ego_yaw - vehicle.get_orientation())
        if check_sum < THRESHOLD_DEGREE or check_sum > 360 - THRESHOLD_DEGREE:
            vehicles_boolean[i] = True
    
    vehicles = vehicles[vehicles_boolean]

    if len(vehicles) == 0:
        return None

    # STEP 3 the lead vehicle will be the nearest vehicle that moves according ego car orientation

    min_distance = math.inf
    nearest_car_index = None
    for i,vehicle in enumerate(vehicles):
        dist = np.subtract(vehicle.get_position(),ego_pos)
        norm = np.linalg.norm(dist)
        if norm < min_distance:
            min_distance = norm
            nearest_car_index = i
    
    return vehicles[nearest_car_index] if nearest_car_index is not None else None
