from shapely.geometry import Polygon,Point

import numpy as np
from math import cos,sin,pi

def compute_bb_verteces_parametrics(p,a,p_orientation,b,b1 = None,sign_x1=1,sign_y1=1,sign_x2=1,sign_y2=1):
    """
        Computes other three bounding box verteces (p1,p2,p3) useful to detect traffic
        light. 
        params:
        p is a bounding box vertex that coincides with car center.
        a lenght of longest bounding box side
        b lenght of shortest bounding box side
        p_orientation car orientation 
        sign_x*, sign_y* can be -1 or 1 and are used to compute points according your specific 
                         coordinate reference system 
    """
    # compute p1 (alias C1)
    x_temp = a * sin(p_orientation) 
    y_temp = a * cos(p_orientation) 
    p1 = [p[0] + sign_x1 * x_temp, p[1] + sign_y1 * y_temp]

    # compute p2 (alias A) 
    x_temp = b * cos(p_orientation) 
    y_temp = b * sin(p_orientation) 

    p2 = [p[0] + sign_x2 * x_temp, p[1] + sign_y2 * y_temp]  

    # Compute p3 (alias B) according the fact that the vector distance between 
    # p0 and p2 is the same between p1 and p3. 
    p3 = [p1[0] + sign_x2 * x_temp, p1[1] + sign_y2 * y_temp] 

    if b1 != None:
        x_temp = b1 * cos(p_orientation) 
        y_temp = b1 * sin(p_orientation)

    # Compute p4 (alias C) that is the opposite of p3 respect p1 (alias C1)
    p4 = [p1[0] - sign_x2 * x_temp, p1[1] - sign_y2 * y_temp] 

    # Compute p5 (alias D) that is the opposite of p2 respect p (aliasi C0)
    p5 = [p[0] - sign_x2 * x_temp, p[1] - sign_y2 * y_temp]  

    return p2,p3,p4,p5 # A,B,C,D

def compute_bb_verteces(p,a,p_orientation,b,b1=None):
    """
    Computes  four bounding box verteces (A,B,C,D).
    Bounding box rapresentation:   

        D-----------C
        |           |    
        p           p1
        |           |
        |           |
        A-----------B

    light params:
        p is a point on segment AD.
        a lenght of longest bounding box side (segment DC or AB)
        b lenght of segment pA (or p1B)
        b1 lenght of segment pD (or p1C)
        p_orientation car orientation 
    """

    # p_orientation = p_orientation*pi/180

    if p_orientation >= 0 and p_orientation <= pi/2:
        p_orientation = pi/2 - p_orientation
        return compute_bb_verteces_parametrics(p,a,p_orientation,b,b1,sign_x2=-1)

    elif p_orientation > pi/2 and p_orientation <= pi:
        p_orientation = p_orientation - pi/2 
        return compute_bb_verteces_parametrics(p,a,p_orientation,b,b1,sign_x1=-1,sign_x2=-1,sign_y2=-1)

    elif p_orientation >= - pi and p_orientation <= -pi/2:
        p_orientation = abs(p_orientation)-pi/2
        return compute_bb_verteces_parametrics(p,a,p_orientation,b,b1, sign_x1=-1,sign_y1=-1,sign_y2=-1)

    elif p_orientation > - pi/2 and p_orientation < 0:
        p_orientation = pi/2 - abs(p_orientation)
        return compute_bb_verteces_parametrics(p,a,p_orientation,b,b1,sign_y1=-1)


def compute_sign(p1,p2,reference_point):
    """
    Computes cross product sign according a reference.
    """
    v1 = np.subtract(reference_point,p1)
    v2 = np.subtract(reference_point,p2)
    return np.sign(np.cross())


def get_stop_wp(waypoints, closest_index,goal_index,traffic_light_pos):
    # note -> this function works only if goal_index - closest_index > 2
    for i in range(closest_index,goal_index):
        dist_wps = np.subtract(waypoints[i+1],waypoints[i])
        s2 = np.add(traffic_light_pos,[dist_wps[1],dist_wps[0]])
        reference_vector = np.subtract(s2,traffic_light_pos)
        v1 = np.subtract(waypoints[i],traffic_light_pos)
        v2 = np.subtract(waypoints[i+1],traffic_light_pos)
        sign_1 = np.sign(np.cross(reference_vector,v1))
        sign_2 = np.sign(np.cross(reference_vector,v2))

        
        if (sign_1 == 0) and pointOnSegment(traffic_light_pos, waypoints[i], s2):
            return i-1
        if (sign_2 == 0) and pointOnSegment(traffic_light_pos, waypoints[i+1], s2):
            return i
        if sign_1 != sign_2:
            return i

    return goal_index



c_0 = [0,0]
teta = 0
a = 3
b1 = 2
b = 1.5

A,B,C,D = compute_bb_verteces(c_0,a,teta,b=b)
print(f"A = {A}\nB = {B}\nC = {C}\nD = {D}")

bb = Polygon([A,B,C,D,A])

# point = Point([2.47,2])
point = Point([1, -6])

print(f"Contains {point} ->",bb.contains(point))
