import matplotlib.pyplot as plt
import numpy as np
from math import cos,sin,pi
point = [0,0]
a = 3
b = 2
teta = 153.43

def compute_semicircle_center_parametric(p,p_orientation,radius,sign_x=1,sign_y=1):
    # compute semicircle center   
    x_temp = radius * cos(p_orientation) 
    y_temp = radius * sin(p_orientation) 
    return [p[0] + sign_x * x_temp, p[1] + sign_y * y_temp]  



def compute_semicircle_center(p,p_orientation,radius):
    """
    Computes other three bounding box verteces (p1,p2,p3) useful to detect traffic
    light params:
        p is a bounding box vertex that coincides with car center.
        p_orientation car orientation 
        radius semicircle radius
    """

    p_orientation = p_orientation*pi/180

    if p_orientation >= 0 and p_orientation <= pi/2:
        p_orientation = pi/2 - p_orientation
        return compute_semicircle_center_parametric(p,p_orientation,radius,sign_y=-1)

    elif p_orientation > pi/2 and p_orientation <= pi:
        p_orientation = p_orientation - pi/2 
        return compute_semicircle_center_parametric(p,p_orientation,radius)

    elif p_orientation >= - pi and p_orientation <= -pi/2:
        p_orientation = abs(p_orientation)-pi/2
        return compute_semicircle_center_parametric(p,p_orientation,radius,sign_x=-1)

    elif p_orientation > - pi/2 and p_orientation < 0:
        p_orientation = pi/2 - abs(p_orientation)
        return compute_semicircle_center_parametric(p,p_orientation,radius,sign_x=-1,sign_y=-1)

def check_point_in_circle(center,point,radius):
    """
    Check if point is inside semicircle with specific center and radius.
    params:
        center: [x,y] semicircle center
        point: [x,y] a point
        radius: semicircle radius 
    return: 
        boolean: true if the point is inside the semicircle
    """

    vector = np.subtract(point,center)
    norm = np.linalg.norm(vector)
    return norm <= radius

def check_opposite_orientation(teta_1,teta_2):
    """
    From the orientation of two moving points verify if they are moving in opposite direction
    params:
        teta_1: orientation (in range [-180,180]) of the first point 
        teta_2: orientation (in range [-180,180]) of the secondo point
    return:
        boolean: true if the direction of the two points is opposite
    """
    THRESHOLD_DEGREE = 10
    REFERENCE_ANGLE = 180
    return abs(REFERENCE_ANGLE-(abs(teta_1)+abs(teta_2))) <= THRESHOLD_DEGREE


def check_traffic_light(ego,traffic_lights,semicircle_radius):
    """
    Checks if a traffic_light is in car trajectory.
    params:
        ego: List([x,y,yaw]) 
        traffic_lights: np.array([id,x,y,yaw],...)
    return:
        int: id of the nearest traffic_light 
    """

    #STEP 1 check traffic_light is in semicircle

    # compute semicircle center
    center = compute_semicircle_center(ego[:2],ego[3],semicircle_radius)

    # first of all verify if there are traffic lights in a circle of specific radius
    # and center  
    vector = np.subtract(traffic_lights[:,1:3],center)
    norm = np.linalg.norm(vector)
    
    index_tl_in_circle = np.where(norm<=semicircle_radius)[0]
    
    # check if there is at least one traffic light
    if len(index_tl_in_circle) == 0:
        return False 
    
    
    tl = traffic_lights[index_tl_in_circle]

    # STEP 2 check if car orientation is opposite to traffic lights orientation
    # this means that car moving towards these traffic lights.
    # Such that car and traffic lights had must be opposite so the sum
    # of their absolute yaw angles must be about 180°

    THRESHOLD_DEGREE = 10
    REFERENCE_ANGLE = 180
    
    # 180 - ( traffic_lights_yaw + car_yaw)
    index_tl_opposite = np.where( np.abs(REFERENCE_ANGLE-(np.abs(tl[:,3])+abs(ego[2]))) <= THRESHOLD_DEGREE)[0]
    if len(index_tl_opposite) == 0:
        return False 

    tl = traffic_lights[index_tl_opposite]

    # STEP3 check if traffic lights are in the upper semicircle  
    ## NOTE MANAGE LIMIT CASE: e.g. car yaw = 0, 180


    dist = np.subtract(traffic_lights[:,1:3],ego[:2])

    # if the car yaw is negative means that it is moving toward
    # traffic light that has y value smaller than car y values,
    #  so the other traffic lights have to be ignored.

  

    # if the car yaw is positive means that it is moving toward
    # traffic light that has y value smaller than car y values
    #  so the other traffic lights have to be ignored.


    # STEP 4 take in account the nearest traffic light

    dist = np.subtract(tl[:,1:3],ego[:2])
    norm = np.linalg.norm(dist,axis = 1)
    tl_index = np.argmin(norm)
    return tl[tl_index] # return the nearest traffic light


    
    

    

    




print(compute_semicircle_center(point,teta,b))

############################################################àà

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
    
####################################################################


