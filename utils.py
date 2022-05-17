import numpy as np
import math


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
    x_temp = a * math.sin(p_orientation) 
    y_temp = a * math.cos(p_orientation) 
    p1 = [p[0] + sign_x1 * x_temp, p[1] + sign_y1 * y_temp]

    # compute p2 (alias A) 
    x_temp = b * math.cos(p_orientation) 
    y_temp = b * math.sin(p_orientation) 

    p2 = [p[0] + sign_x2 * x_temp, p[1] + sign_y2 * y_temp]  

    # Compute p3 (alias B) according the fact that the vector distance between 
    # p0 and p2 is the same between p1 and p3. 
    p3 = [p1[0] + sign_x2 * x_temp, p1[1] + sign_y2 * y_temp] 

    if b1 != None:
        x_temp = b1 * math.cos(p_orientation) 
        y_temp = b1 * math.sin(p_orientation)

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

    if p_orientation >= 0 and p_orientation <= math.pi/2:
        p_orientation = math.pi/2 - p_orientation
        return compute_bb_verteces_parametrics(p,a,p_orientation,b,b1,sign_x2=-1)

    elif p_orientation > math.pi/2 and p_orientation <= math.pi:
        p_orientation = p_orientation - math.pi/2 
        return compute_bb_verteces_parametrics(p,a,p_orientation,b,b1,sign_x1=-1,sign_x2=-1,sign_y2=-1)

    elif p_orientation >= - math.pi and p_orientation <= -math.pi/2:
        p_orientation = abs(p_orientation)-math.pi/2
        return compute_bb_verteces_parametrics(p,a,p_orientation,b,b1, sign_x1=-1,sign_y1=-1,sign_y2=-1)

    elif p_orientation > - math.pi/2 and p_orientation < 0:
        p_orientation = math.pi/2 - abs(p_orientation)
        return compute_bb_verteces_parametrics(p,a,p_orientation,b,b1,sign_y1=-1)


def compute_point_along_direction_parametric(p,p_orientation,a,sign_x=1,sign_y=1):
    x_temp = a * math.sin(p_orientation) 
    y_temp = a * math.cos(p_orientation) 
    return [p[0] + sign_x * x_temp, p[1] + sign_y * y_temp]


def compute_point_along_direction(start_point,direction,distance):
    """
    Es:
                                 distance
                    new_point <----------------- start_point
                                 direction = - pi
    """
    p_orientation = direction
    if p_orientation >= 0 and p_orientation <= math.pi/2:
        p_orientation = math.pi/2 - p_orientation
        return compute_point_along_direction_parametric(start_point,p_orientation,distance)

    elif p_orientation > math.pi/2 and p_orientation <= math.pi:
        p_orientation = p_orientation - math.pi/2 
        return compute_point_along_direction_parametric(start_point,p_orientation,distance,sign_x=-1)

    elif p_orientation >= - math.pi and p_orientation <= -math.pi/2:
        p_orientation = abs(p_orientation)-math.pi/2
        return compute_point_along_direction_parametric(start_point,p_orientation,distance, sign_x=-1,sign_y=-1)

    elif p_orientation > - math.pi/2 and p_orientation < 0:
        p_orientation = math.pi/2 - abs(p_orientation)
        return compute_point_along_direction_parametric(start_point,p_orientation,distance,sign_y=-1)
