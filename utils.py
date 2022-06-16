import math
import numpy as np

def compute_bb_verteces_parametrics(p,a,p_orientation,b,b1 = None,sign_x1=1,sign_y1=1,sign_x2=1,sign_y2=1):
    """
        Computes other three bounding box verteces (p1,p2,p3) useful to detect traffic
        light. 
        params:
        p list([x,y]): is a point on segment AD.
        a float: lenght of longest bounding box side (segment DC or AB)
        b float: lenght of segment pA (or p1B)
        b1 float: lenght of segment pD (or p1C) (if None it will be eqaul to b)
        p_orientation float: orientation in radians (-pi to pi) needed to define segment AB from point p
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

    params:
        p list([x,y]): is a point on segment AD.
        a float: lenght of longest bounding box side (segment DC or AB)
        p_orientation float: orientation in radians (-pi to pi) needed to define segment AB from point p
        b float: lenght of segment pA (or p1B)
        b1 float: lenght of segment pD (or p1C) (if None it will be eqaul to b)
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
    """
    Fixed a point and orientation derives an other point along a segment that starts from point p with lenght
    a and orientation p_orientation
    params:     
        p list([x,y]): is a point on segment AD.
        a float: distance from point p to new point (segment DC or AB)
        p_orientation float: orientation in radians (-pi to pi) needed to define position of new points respect to point p
        sign_x, sign_y can be -1 or 1 and are used to compute points according your specific 
                         coordinate reference system 
    """
    x_temp = a * math.sin(p_orientation) 
    y_temp = a * math.cos(p_orientation) 
    return [p[0] + sign_x * x_temp, p[1] + sign_y * y_temp]


def compute_point_along_direction(start_point,direction,distance):
    """
    Fixed a start point and orientation derives an other point along a segment that starts from point p with lenght
    equal to distance and orientation eqaul to direction
    params:     
        p list([x,y]): is a point on segment AD.
        distance float: distance from point start_point to new point (segment DC or AB)
        direction float: orientation in radians (-pi to pi) needed to define position of new points respect to point p    
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


def compute_middle_point(x, y, width, height):
    x_middle_point= x + width//2
    y_middle_point= y + height
    middle_point = (x_middle_point, y_middle_point)
    
    return middle_point


# Compute the waypoint index that is closest to the ego vehicle, and return
# it as well as the distance from the ego vehicle to that waypoint.
def get_closest_index(waypoints, ego_state):
    """Gets closest index a given list of waypoints to the vehicle position.

    args:
        waypoints: current waypoints to track. (global frame)
            length and speed in m and m/s.
            (includes speed to track at each x,y location.)
            format: [[x0, y0, v0],
                     [x1, y1, v1],
                     ...
                     [xn, yn, vn]]
            example:
                waypoints[2][1]: 
                returns the 3rd waypoint's y position

                waypoints[5]:
                returns [x5, y5, v5] (6th waypoint)
        ego_state: ego state vector for the vehicle. (global frame)
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)

    returns:
        [closest_len, closest_index]:
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
    """
    closest_len = float('Inf')
    closest_index = 0

    for i in range(len(waypoints)):
        temp = (waypoints[i][0] - ego_state[0])**2 + (waypoints[i][1] - ego_state[1])**2
        if temp < closest_len:
            closest_len = temp
            closest_index = i
    closest_len = np.sqrt(closest_len)

    return closest_len, closest_index

# Checks if p2 lies on segment p1-p3, if p1, p2, p3 are collinear.        
def pointOnSegment(p1, p2, p3):
    if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and \
       (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
        return True
    else:
        return False

def get_stop_wp(waypoints, closest_index,goal_index,position_to_stop):
    # note -> this function works only if goal_index - closest_index > 2
    for i in range(closest_index,goal_index):
        dist_wps = np.subtract(waypoints[i+1][:2],waypoints[i][:2])
        s2 = np.add(position_to_stop,[dist_wps[1],dist_wps[0]])
        reference_vector = np.subtract(s2,position_to_stop)
        v1 = np.subtract(waypoints[i][:2],position_to_stop)
        v2 = np.subtract(waypoints[i+1][:2],position_to_stop)
        sign_1 = np.sign(np.cross(reference_vector,v1))
        sign_2 = np.sign(np.cross(reference_vector,v2))

        
        if (sign_1 == 0) and pointOnSegment(position_to_stop, waypoints[i][:2], s2):
            return i-1
        if (sign_2 == 0) and pointOnSegment(position_to_stop, waypoints[i+1][:2], s2):
            return i
        if sign_1 != sign_2:
            return i

    return goal_index

