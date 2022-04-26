import matplotlib.pyplot as plt
import numpy as np
from math import cos,sin,pi
point = [0,0]
a = 3
b = 2
teta = -30



def compute_bb_verteces_parametrics(p,a,b,p_orientation,sign_x1=1,sign_y1=1,sign_x2=1,sign_y2=1):
    """
        Computes other three bounding box verteces (p1,p2,p3) useful to detect traffic
        light. 
        params:
        p is a bounding box vertex that coincides with car center.
        a lenght of longest bounding box side
        b lenght of shortest bounding box side
        p_orientation car orientation 
        sign_x*, sign_y* can be -1 or 1 and are used to compute points according your specific 
                         orientation 
    """
    # compute p1
    x_temp = a * sin(p_orientation) 
    y_temp = a * cos(p_orientation) 
    p1 = [p[0] + sign_x1 * x_temp, p[1] + sign_y1 * y_temp]

    # compute p2  
    x_temp = b * cos(p_orientation) 
    y_temp = b * sin(p_orientation) 

    p2 = [p[0] + sign_x2 * x_temp, p[1] + sign_y2 * y_temp]  

    # Compute p3 according the fact that the vector distance between 
    # p0 and p2 is the same between p1 and p3. 
    p3 = [p1[0] + sign_x2 * x_temp, p1[1] + sign_y2 * y_temp] 

    return p1,p2,p3

def compute_bb_verteces(p,a,b,p_orientation):
    """
    Computes other three bounding box verteces (p1,p2,p3) useful to detect traffic
    light params:
        p is a bounding box vertex that coincides with car center.
        a lenght of longest bounding box side
        b lenght of shortest bounding box side
        p_orientation car orientation 
    """

    p_orientation = p_orientation*pi/180

    if p_orientation >= 0 and p_orientation <= pi/2:
        p_orientation = pi/2 - p_orientation
        return compute_bb_verteces_parametrics(p,a,b,p_orientation,sign_y2=-1)

    elif p_orientation > pi/2 and p_orientation <= pi:
        p_orientation = p_orientation - pi/2 
        return compute_bb_verteces_parametrics(p,a,b,p_orientation,sign_x1=-1)

    elif p_orientation >= - pi and p_orientation <= -pi/2:
        p_orientation = abs(p_orientation)-pi/2
        return compute_bb_verteces_parametrics(p,a,b,p_orientation, sign_x1=-1,sign_y1=-1,sign_x2=-1)

    elif p_orientation > - pi/2 and p_orientation < 0:
        p_orientation = pi/2 - abs(p_orientation)
        return compute_bb_verteces_parametrics(p,a,b,p_orientation,sign_y1=-1,sign_x2=-1,sign_y2=-1)

def compute_bb_vertex_first_quadrant(p,a,b,p_orientation):
    """
        This function is used when the point p1 is in first quadrant.
        Computes other three bounding box verteces (p1,p2,p3) useful to detect traffic
        light. 
        params:
        p is a bounding box vertex that coincides with car center.
        a lenght of longest bounding box side
        b lenght of shortest bounding box side
        p_orientation car orientation 
    """
    # compute p1
    x_temp = a * cos(p_orientation) 
    y_temp = a * sin(p_orientation) 
    p1 = [p[0] + x_temp, p[1] + y_temp]

    # compute p2 using consideration according the two parallel lines
    # crossed by a transversal theorem
    
    x_temp = b * sin(p_orientation) 
    y_temp = b * cos(p_orientation) 

    p2 = [p[0] + x_temp, p[1] - y_temp]  

    # Compute p3 according the fact that the vector distance between 
    # p0 and p2 is the same between p1 and p3. 
    p3 = [p1[0] + x_temp, p1[1] - y_temp] 

    return p1,p2,p3

def compute_bb_vertex_second_quadrant(p,a,b,p_orientation):
    """
        This function is used when the point p1 is in second quadrant.
        Computes other three bounding box verteces (p1,p2,p3) useful to detect traffic
        light. 
        params:
        p is a bounding box vertex that coincides with car center.
        a lenght of longest bounding box side
        b lenght of shortest bounding box side
        p_orientation car orientation 
    """
    # compute p1
    x_temp = a * cos(p_orientation) 
    y_temp = a * sin(p_orientation) 
    p1 = [p[0] + x_temp, p[1] + y_temp]

    # compute p2 using trigonometry rule 
    p_orientation -= pi/2
    x_temp = b * cos(p_orientation) 
    y_temp = b * sin(p_orientation) 

    p2 = [p[0] + x_temp, p[1] + y_temp]  

    # Compute p3 according the fact that the vector distance between 
    # p0 and p2 is the same between p1 and p3. 
    p3 = [p1[0] + x_temp, p1[1] + y_temp] 

    return p1,p2,p3

def compute_bb_vertex_third_quadrant(p,a,b,p_orientation):
    """
        This function is used when the point p1 is in third quadrant.
        Computes other three bounding box verteces (p1,p2,p3) useful to detect traffic
        light. 
        params:
        p is a bounding box vertex that coincides with car center.
        a lenght of longest bounding box side
        b lenght of shortest bounding box side
        p_orientation car orientation 
    """
    # in this way p_orientation is used to compute
    # both p1 and p2
    p_orientation = abs(p_orientation)-pi/2

    # compute p1
    x_temp = a * sin(p_orientation) 
    y_temp = a * cos(p_orientation) 
    p1 = [p[0] - x_temp, p[1] - y_temp]

    # compute p2  
    x_temp = b * cos(p_orientation) 
    y_temp = b * sin(p_orientation) 

    p2 = [p[0] - x_temp, p[1] + y_temp]  

    # Compute p3 according the fact that the vector distance between 
    # p0 and p2 is the same between p1 and p3. 
    p3 = [p1[0] - x_temp, p1[1] + y_temp] 

    return p1,p2,p3

def compute_bb_vertex_fourth_quadrant(p,a,b,p_orientation):
    """
        This function is used when the point p1 is in fourth quadrant.
        Computes other three bounding box verteces (p1,p2,p3) useful to detect traffic
        light. 
        params:
        p is a bounding box vertex that coincides with car center.
        a lenght of longest bounding box side
        b lenght of shortest bounding box side
        p_orientation car orientation 
    """
    # in this way p_orientation is used to compute
    # both p1 and p2
    p_orientation = pi/2 - abs(p_orientation)

    # compute p1
    x_temp = a * sin(p_orientation) 
    y_temp = a * cos(p_orientation) 
    p1 = [p[0] + x_temp, p[1] - y_temp]

    # compute p2  
    x_temp = b * cos(p_orientation) 
    y_temp = b * sin(p_orientation) 

    p2 = [p[0] - x_temp, p[1] - y_temp]  

    # Compute p3 according the fact that the vector distance between 
    # p0 and p2 is the same between p1 and p3. 
    p3 = [p1[0] - x_temp, p1[1] - y_temp] 

    return p1,p2,p3

def compute_bb_vertex(p,a,b,p_orientation):
    p_orientation = p_orientation*pi/180
    if p_orientation >= 0 and p_orientation <= pi/2:
        return compute_bb_vertex_first_quadrant(p,a,b,p_orientation)
    elif p_orientation > pi/2 and p_orientation <= pi:
        #p_orientation = pi - p_orientation 
        return compute_bb_vertex_second_quadrant(p,a,b,p_orientation)
    elif p_orientation >= - pi and p_orientation <= -pi/2:
        return compute_bb_vertex_third_quadrant(p,a,b,p_orientation)
    elif p_orientation > - pi/2 and p_orientation < 0:
     return compute_bb_vertex_fourth_quadrant(p,a,b,p_orientation)

def check_traffic_light_presence(p0,p1,p2,p3,tl_point):
    """
        p0 ########## p1
           #        #
           #        #
        p2 ########## p3
    """

    points = [p0,p1,p2,p3]
    points = np.array(points)
    

print(compute_bb_verteces(point,a,b,teta))

