import numpy as np
import math
from utils import *
from shapely.geometry import Point, Polygon


def check_traffic_light(ego_pos,ego_yaw,traffic_lights,lookahead,looksideways_right,looksideways_left=None):
    """
    Checks if a traffic_light is in car trajectory.
    params:
        ego_pos: List([x,y]) 
        ego_yaw: angle -pi to pi
        traffic_lights: np.array([id,x,y,yaw],...)
        lookahead: ahead view of veichle
        looksideways_right: right lateral view of vehicle
        looksideways_left: left lateral view of vehicle

    return:
        int: if detected return traffic light info as list, else empty list
    """
    #STEP 1 check traffic_light is in bounding box
    # compute bounding box starting from car bonnet
    center_point = compute_point_along_direction(ego_pos,ego_yaw,distance=2)
        
    A,B,C,D = compute_bb_verteces(center_point,lookahead,ego_yaw,looksideways_right,looksideways_left)
    bb = Polygon([A,B,C,D,A])
    n_tl = len(traffic_lights)
    index_tl_in_bb = np.array(np.zeros((n_tl,)),dtype=bool)
    for i,tl in enumerate(traffic_lights):
        # x,y traffic light
        tl_point = Point(tl[1:3])
        index_tl_in_bb[i] = bb.contains(tl_point)

    # check if there is at least one traffic light

    tl = traffic_lights[index_tl_in_bb]
   

    if len(tl)==0:
        return []
    # STEP 2 check if car orientation is opposite to traffic lights orientation
    # this means that car moving towards these traffic lights.
    # Such that car and traffic lights had must be opposite so the sum
    # of their absolute yaw angles must be about 270° or 90°

    THRESHOLD_DEGREE = 3
    
    ego_yaw = ego_yaw*180/math.pi
    check_sum_90 = np.abs(90-np.abs(tl[:,3]+ego_yaw))<=THRESHOLD_DEGREE
    check_sum_270 = np.abs(270-np.abs(tl[:,3]+ego_yaw))<=THRESHOLD_DEGREE
    check = np.logical_or(check_sum_90,check_sum_270)
    tl = tl[check] # This line allows to filter elements that satisfy "check" condition (that's a boolean array)
    

    
    if len(tl) == 0:
        return [] 


    # STEP 3 take in account the nearest traffic light according vehicle position

    dist = np.subtract(tl[:,1:3],ego_pos)
    norm = np.linalg.norm(dist,axis = 1)
    index_tl = np.argmin(norm)


    return tl[index_tl] # return the nearest traffic light
