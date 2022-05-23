import numpy as np
import math
from pedestrians import CAR_LATERAL_MARGIN
from utils import *
from shapely.geometry import Point, Polygon, LineString


def detect_lead_vehicle(ego_pos,ego_yaw,vehicles,lookahead,looksideways_right=2.5,looksideways_left=1.5):
    """
        Detects the presence of a vehicle to follow on the ego trajectory. If a vehicle is detected returns it
        params:
            ego_pos list([x,y]):  car coordinates
            ego_yaw float : car orientation
            vehicles np.array(): array of vehicles
            lookahead float:  max  look a head distance 
            looksideways_right float: max look right distance 
            looksideways_left float: max looksideways left distance
    """
    if len(vehicles)==0:
        return None

    # Step 1 filter vehicles in bb
    A,B,C,D = compute_bb_verteces(ego_pos,lookahead,ego_yaw,looksideways_right,looksideways_left)
    bb = Polygon([A,B,C,D,A])

    vehicles_boolean = vehicles == None
    for i,vehicle in enumerate(vehicles):
        vehicle_bb_verteces = vehicle.get_bounding_box()

        for bb_vertex in vehicle_bb_verteces:
            vertex = Point(bb_vertex)
            if bb.contains(vertex):
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



def check_vehicles(ego_pos,ego_yaw,vehicles,lookahead,looksideways_right,looksideways_left,waypoints,closest_index,goal_index,lead_vehicle=True):
    """
        Detects the presence of a vehicle on the ego trajectory. If a vehicle collision is estimated returns 
        true flag and the new goal index to stop.
        params:
            ego_pos list([x,y]):  car coordinates
            ego_yaw float : car orientation
            vehicles np.array(): array of vehicles
            lookahead float:  max  look a head distance 
            looksideways_right float: max look right distance 
            looksideways_left float: max looksideways left distance
            waypoints np.array(): point that car should follow
            closest_index int:  the nearest waypoint index
            goal_index int: the goal waypoint index
            lead_vehicle Boolean: presence of a vehicle to follow 
    """
    if len(vehicles)==0:
        return False,goal_index

    # Step 1 filter pedetrians in bb
    A,B,C,D = compute_bb_verteces(ego_pos,lookahead,ego_yaw,looksideways_right,looksideways_left)
    bb = Polygon([A,B,C,D,A])
    
    # numpy array pedestrians
    
    vehicles_boolean = vehicles == None
    for i,v in enumerate(vehicles):
        vehicles_bb_verteces = v.get_bounding_box()
        for bb_vertex in vehicles_bb_verteces:
            
            vertex = Point(bb_vertex)
            if bb.contains(vertex):
                vehicles_boolean[i] = True
                break
        
    # considered only pedestrians inside bounding box
    vehicles = vehicles[vehicles_boolean]

    # STEP 2 in the case there is no lead vehicle is in the scene we check only for cars in direction 
    # discording to us (ONLY in FOLLOW_LANE STATE) 
    # because if we detect a person and no lead car is just detected 

    if lead_vehicle: 

        THRESHOLD_DEGREE = 3.5
        
        ego_yaw = ego_yaw*180/math.pi
        vehicles_boolean = vehicles == None

        for i, vehicle in enumerate(vehicles):
            check_sum = abs(ego_yaw - vehicle.get_orientation())
            if check_sum > THRESHOLD_DEGREE and check_sum < 360 - THRESHOLD_DEGREE:
                vehicles_boolean[i] = True
        
        vehicles = vehicles[vehicles_boolean]
    
    start_point = ego_pos
    intersected = False
    index = closest_index

    # plus one so in this way also goal index is used to check intersection
    for index in range(closest_index,goal_index+1):
        next_point = waypoints[index][:2]

        v_diff = np.subtract(next_point,start_point)
        norm = np.linalg.norm(v_diff)
        orientation = math.atan2(v_diff[1],v_diff[0])
        car_extent_y = 1.5
        A,B,C,D = compute_bb_verteces(start_point,norm,orientation,car_extent_y+CAR_LATERAL_MARGIN)
        car_path = Polygon([A,B,C,D,A])
        #car_path = LineString([start_point,next_point])
        v_distance = 15 # in further work udapte this
        for v in vehicles:
            v_start_point = v.get_position()
            v_orientation = v.get_orientation()*math.pi/180
            v_next_point = compute_point_along_direction(v_start_point,v_orientation,v_distance)
            v_path = LineString([v_start_point,v_next_point])
            intersected = v_path.intersects(car_path)
            if intersected:
                return intersected,closest_index
        start_point = next_point

    return intersected,index