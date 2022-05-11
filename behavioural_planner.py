#!/usr/bin/env python3
from cv2 import distanceTransformWithLabels
import numpy as np
import math
from shapely.geometry import Point, Polygon

# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2

STATES = ["FOLLOW_LANE","DECELERATE_TO_STOP","STAY_STOPPED"]
# Stop speed threshold
STOP_THRESHOLD = 0.5


# traffic light status
GREEN = 0
YELLOW = 1
RED = 2

# semicircle radius used to detect traffic lights
RADIUS = 50 # metres

BASE_LOOKSIDEWAYS_RIGHT = 3
BASE_LOOKSIDEWAYS_LEFT = 3

MAX_PEDESTRIAN_LOOKSIDEWAYS_LEFT = 4
MAX_PEDESTRIAN_LOOKSIDEWAYS_RIGHT = 3.15


class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead,traffic_lights,tl_dict,pedestrians=[], vehicles = []):
        self._lookahead                     = lookahead
        self._follow_lead_vehicle_lookahead = lead_vehicle_lookahead
        self._state                         = FOLLOW_LANE
        self._follow_lead_vehicle           = False
        self._obstacle_on_lane              = False
        self._goal_state                    = [0.0, 0.0, 0.0]
        self._goal_index                    = 0
        self._stop_count                    = 0
        self._lookahead_collision_index     = 0
        self._traffic_lights                = traffic_lights
        self._tl_dict                       = tl_dict
        #self._tl_id                         = None
        self._current_traffic_light         = []
        self._pedestrians                   = pedestrians
        self._vehicles                      = vehicles
        self._lead_vehicle                  = None
        self._vehicles_dict                 = None
        self._pedestrian_detected           = False


    def set_lookahead(self, lookahead):
        self._lookahead = lookahead

    def set_tl_dict(self,tl_dict):
        self._tl_dict = tl_dict

    def set_pedestrians(self,pedestrians):
        self._pedestrians  = pedestrians

    def set_vehicles(self,vehicles):
        self._vehicles = vehicles

    def set_vehicles_dict(self,vehicles_dict):
        self._vehicles_dict = vehicles_dict

    
    

    #Handles state transitions and computes the goal state.
    def transition_state(self, waypoints, ego_state, closed_loop_speed):
        """Handles state transitions and computes the goal state.  
        
        args:
            waypoints: current waypoints to track (global frame). 
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
            closed_loop_speed: current (closed-loop) speed for vehicle (m/s)
        variables to set:
            self._goal_index: Goal index for the vehicle to reach
                i.e. waypoints[self._goal_index] gives the goal waypoint
            self._goal_state: Goal state for the vehicle to reach (global frame)
                format: [x_goal, y_goal, v_goal]
            self._state: The current state of the vehicle.
                available states: 
                    FOLLOW_LANE         : Follow the global waypoints (lane).
                    DECELERATE_TO_STOP  : Decelerate to stop.
                    STAY_STOPPED        : Stay stopped.
            self._stop_count: Counter used to count the number of cycles which
                the vehicle was in the STAY_STOPPED state so far.
        useful_constants:
            STOP_THRESHOLD  : Stop speed threshold (m). The vehicle should fully
                              stop when its speed falls within this threshold.
            STOP_COUNTS     : Number of cycles (simulation iterations) 
                              before moving from stop sign.
        """
        # In this state, continue tracking the lane by finding the
        # goal index in the waypoint list that is within the lookahead
        # distance. Then, check to see if the waypoint path intersects
        # with any stop lines. If it does, then ensure that the goal
        # state enforces the car to be stopped before the stop line.
        # You should use the get_closest_index(), get_goal_index(), and
        # check_for_stop_signs() helper functions.
        # Make sure that get_closest_index() and get_goal_index() functions are
        # complete, and examine the check_for_stop_signs() function to
        # understand it.
        closest_index = None
        pedestrian_looksideways_left = min(BASE_LOOKSIDEWAYS_LEFT+closed_loop_speed/1.8,MAX_PEDESTRIAN_LOOKSIDEWAYS_LEFT)
        pedestrian_looksideways_right = min(BASE_LOOKSIDEWAYS_RIGHT+closed_loop_speed/1.8,MAX_PEDESTRIAN_LOOKSIDEWAYS_RIGHT)
        
        # 2.5 is the speed that car try to follow when it is making a turn
        pedestrian_lookahead = self._lookahead
        separation_distance = self._lookahead 

        #print("[BP.transition_state] pedestrian lookahead",pedestrian_lookahead)


        if self._state == FOLLOW_LANE:
            # First, find the closest index to the ego vehicle.
            closest_len, closest_index = get_closest_index(waypoints, ego_state)
            
            # Next, find the goal index that lies within the lookahead distance
            # along the waypoints.
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)


            wp = [ waypoints[goal_index][0], waypoints[goal_index][1],waypoints[goal_index][2]]      

            wp_speed = waypoints[goal_index][2]


            if not self._follow_lead_vehicle:
                # If no lead vehicle is already known search for a lead vehicle to follow

                self._lead_vehicle = detect_lead_vehicle(ego_state[:2],ego_state[2],self._vehicles,separation_distance)
               
                # if a lead vehicle is detected set true follow_lead_vehicle flag
                if self._lead_vehicle is not None:
                    self._follow_lead_vehicle = True
            else:
                # If a lead vehicle is known checks if it is still a lead vehicle 
                id = self._lead_vehicle.get_id()
                self._lead_vehicle = self._vehicles_dict[id]
                self._follow_lead_vehicle_lookahead = separation_distance
                self._lead_vehicle = detect_lead_vehicle(ego_state[:2],ego_state[2],np.array([self._lead_vehicle],dtype=object),separation_distance)

                #self.check_for_lead_vehicle(ego_state, self._lead_vehicle.get_position())
                if self._lead_vehicle is  None:
                    self._follow_lead_vehicle = False

           

            goal_index_pd = goal_index
            goal_index_tl = goal_index

            ### check pedestrian intersection
            pedestrain_detected, car_stop = check_pedestrian(ego_state[:2],ego_state[2],closed_loop_speed,
                self._pedestrians,lookahead= pedestrian_lookahead,looksideways_right=pedestrian_looksideways_right,looksideways_left=pedestrian_looksideways_left)
            # pedestrain_detected, car_stop = check_pedestrian(ego_state[:2],ego_state[2],closed_loop_speed,
            #     self._pedestrians,self._lookahead,looksideways_right=2.5,looksideways_left=4)
                
            self._pedestrian_detected = pedestrain_detected
            if pedestrain_detected:
                #print("[BP.trasistion_state] pedestrian_detected")
                if closed_loop_speed > STOP_THRESHOLD:
                    goal_index_pd = get_stop_wp(waypoints,closest_index,goal_index,car_stop)
                wp_speed = 0
                self._state = DECELERATE_TO_STOP
                       
            self._current_traffic_light = check_traffic_light(ego_state[:2],ego_state[2],self._traffic_lights,self._lookahead,looksideways_right=4.5)
            status = None

            traffic_light_on_path = len(self._current_traffic_light)>0
            if traffic_light_on_path:

                id = self._current_traffic_light[0]
                status = self._tl_dict[id]     
                if status != GREEN:
                    stop = True
                    if status == YELLOW:
                        dist = np.subtract(ego_state[:2],self._current_traffic_light[1:3])
                        norm = np.linalg.norm(dist)
                        # if distance from traffic light is greater than 1 meter and it is in YELLOW STATE you must stop. 
                        stop = norm > 1  

                    if stop:
                        goal_index_tl = get_stop_wp(waypoints,closest_index,goal_index,self._current_traffic_light[1:3])
                        wp_speed = 0
                        self._state = DECELERATE_TO_STOP
            
            # define goal index according the fact that a pedestrain could be located nearest to the car than
            # the traffic light or viceversa.
            goal_index = goal_index_pd if goal_index_pd < goal_index_tl else goal_index_tl

            wp = [waypoints[goal_index][0],waypoints[goal_index][1],wp_speed]
            self._goal_index = goal_index
            self._goal_state = wp                  

        
        elif self._state == DECELERATE_TO_STOP:

            if abs(closed_loop_speed) <= STOP_THRESHOLD:
                self._state = STAY_STOPPED
                return
            
            

            #######################
            ## Vehicle collision ##
            #######################

            
            closest_len, closest_index = get_closest_index(waypoints, ego_state)
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)

            # if new goal_index is greater than last goal_index we don't update the current goal_index
            # because in this state the aim is to decelerate and stop the car.
            # we update the current goal index if the new goal index is smaller or  the latest goal_index
            # is smaller than the actual closest_index. The last condition means that the car passed the latest goal_index and so
            # the new stop goal index will be the closest_index.    
            if goal_index>self._goal_index:
                #if goal_index<closest_index:
                goal_index = self._goal_index


            # we chose the goal index to stop according the fact that         
            goal_index_pd = goal_index
            goal_index_tl = goal_index
            
            pedestrain_detected, car_stop = check_pedestrian(ego_state[:2],ego_state[2],closed_loop_speed,self._pedestrians,lookahead=pedestrian_lookahead,looksideways_right=pedestrian_looksideways_right,looksideways_left=pedestrian_looksideways_left)
            self._pedestrian_detected = pedestrain_detected

            
            if pedestrain_detected:
                #print("[BP.trasistion_state] pedestrian_detected")
                
                goal_index_pd = get_stop_wp(waypoints,closest_index,goal_index,car_stop)
                
                #wp = [ waypoints[goal_index][0],waypoints[goal_index][1],0]
                #self._goal_index = goal_index
                #self._goal_state = wp
            
            traffic_light_on_path = len(self._current_traffic_light)>0
            if traffic_light_on_path:

                id = self._current_traffic_light[0]

                if self._tl_dict[id] != GREEN :
                    tl_position = self._current_traffic_light[1:3]
                    goal_index_tl = get_stop_wp(waypoints,closest_index,goal_index,tl_position)
                elif self._tl_dict[id] == GREEN and not pedestrain_detected:
                        self._state = FOLLOW_LANE
                        self._current_traffic_light = []
                        return

            # this condition is when the car has previusly detected a pedestrain on the road
            # and than this pedestrian goes out of road. 
            if not pedestrain_detected and not traffic_light_on_path:
                self._state = FOLLOW_LANE
                return

           

            # define goal index according the fact that a pedestrain could be located nearest to the car than
            # the traffic light or viceversa
            goal_index = goal_index_pd if goal_index_pd < goal_index_tl else goal_index_tl

            wp = [ waypoints[goal_index][0],waypoints[goal_index][1],0]
            self._goal_index = goal_index
            self._goal_state = wp

              
            

        
        elif self._state == STAY_STOPPED:

            # cehck if there are some pedetrian along car trajectory
            check_pedestrian_collision, car_stop = check_pedestrian(ego_state[:2],ego_state[2],closed_loop_speed,
            self._pedestrians,lookahead=self._lookahead,looksideways_right=BASE_LOOKSIDEWAYS_RIGHT,looksideways_left=BASE_LOOKSIDEWAYS_LEFT)
            
            self._pedestrian_detected = check_pedestrian_collision

            

            # check if there is a red traffic light 
            traffic_light_on_path = len(self._current_traffic_light)>0
            traffic_light_stop = False
            if traffic_light_on_path:
                id = self._current_traffic_light[0]
                traffic_light_stop = self._tl_dict[id] != GREEN


            #traffic_light_condition = True if self._tl_id is None else self._tl_dict[self._tl_id] == GREEN

            # if no pedetrain and red traffic light are along car trajectory go to FOLLOW_LANE
            if not check_pedestrian_collision and not traffic_light_stop:
                self._state = FOLLOW_LANE
                # reset current traffic light status
                self._current_traffic_light = []
           
            
           
        else:
            raise ValueError('Invalid state value.')
        

  
    def get_goal_index(self, waypoints, ego_state, closest_len, closest_index):
        """Gets the goal index for the vehicle. 
        
        Set to be the earliest waypoint that has accumulated arc length
        accumulated arc length (including closest_len) that is greater than or
        equal to self._lookahead.

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
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        returns:
            wp_index: Goal index for the vehicle to reach
                i.e. waypoints[wp_index] gives the goal waypoint
        """
        # Find the farthest point along the path that is within the
        # lookahead distance of the ego vehicle.
        # Take the distance from the ego vehicle to the closest waypoint into
        # consideration.
        arc_length = closest_len
        wp_index = closest_index

        # In this case, reaching the closest waypoint is already far enough for
        # the planner.  No need to check additional waypoints.
        if arc_length > self._lookahead:
            return wp_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            return wp_index

        # Otherwise, find our next waypoint.
        while wp_index < len(waypoints) - 1:
            arc_length += np.sqrt((waypoints[wp_index][0] - waypoints[wp_index+1][0])**2 + (waypoints[wp_index][1] - waypoints[wp_index+1][1])**2)
            if arc_length > self._lookahead: break
            wp_index += 1

        return wp_index % len(waypoints)
                
    # Checks to see if we need to modify our velocity profile to accomodate the
    # lead vehicle.
    def check_for_lead_vehicle(self, ego_state, lead_car_position):
        """Checks for lead vehicle within the proximity of the ego car, such
        that the ego car should begin to follow the lead vehicle.

        args:
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            lead_car_position: The [x, y] position of the lead vehicle.
                Lengths are in meters, and it is in the global frame.
        sets:
            self._follow_lead_vehicle: Boolean flag on whether the ego vehicle
                should follow (true) the lead car or not (false).
        """
        # Check lead car position delta vector relative to heading, as well as
        # distance, to determine if car should be followed.
        # Check to see if lead vehicle is within range, and is ahead of us.
        if not self._follow_lead_vehicle:
            # Compute the angle between the normalized vector between the lead vehicle
            # and ego vehicle position with the ego vehicle's heading vector.
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0], 
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)
            # In this case, the car is too far away.   
            if lead_car_distance > self._follow_lead_vehicle_lookahead:
                return

            lead_car_delta_vector = np.divide(lead_car_delta_vector, 
                                              lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), 
                                  math.sin(ego_state[2])]
            # Check to see if the relative angle between the lead vehicle and the ego
            # vehicle lies within +/- 45 degrees of the ego vehicle's heading.
            if np.dot(lead_car_delta_vector, 
                      ego_heading_vector) < (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = True

        else:
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0], 
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)

            # Check to see if the lead vehicle is still within the ego vehicle's
            # frame of view.
            lead_car_delta_vector = np.divide(lead_car_delta_vector, lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), math.sin(ego_state[2])]

            is_in_car_view = np.dot(lead_car_delta_vector, ego_heading_vector) > (1 / math.sqrt(2))

            if is_in_car_view:
                if lead_car_distance < self._follow_lead_vehicle_lookahead + 15:
                    return
                
            # if np.dot(lead_car_delta_vector, ego_heading_vector) > (1 / math.sqrt(2)):
            #     return

            # # Add a 15m buffer to prevent oscillations for the distance check.
            # if lead_car_distance < self._follow_lead_vehicle_lookahead + 15:
            #     return
            

            self._follow_lead_vehicle = False

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

    # p_orientation = p_orientation*pi/180

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
    # 180 - ( traffic_lights_yaw + car_yaw)
    #index_tl_opposite = np.where( np.abs(REFERENCE_ANGLE-(np.abs(tl[:,3])+abs(ego_yaw))) <= THRESHOLD_DEGREE)[0]
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

def check_pedestrian(ego_pos,ego_yaw,ego_speed,pedestrians,lookahead,looksideways_right,looksideways_left):
    
    if len(pedestrians)==0:
        return False,[]

    # Step 1 filter pedetrians in bb
    A,B,C,D = compute_bb_verteces(ego_pos,lookahead,ego_yaw,looksideways_right,looksideways_left)
    bb = Polygon([A,B,C,D,A])
    
    # numpy array pedestrians
    pds  = pedestrians
    pds_boolean = pds == None
    flag = False
    for i,pd in enumerate(pds):
        # get pedestrian bb
        pedestrian_bb_verteces = pd[0]
        

        for bb_vertex in pedestrian_bb_verteces:
            
            vertex = Point(bb_vertex)
            if bb.contains(vertex):
                flag = True
                pds_boolean[i] = True
                break
        
      
    # considered only pedestrians inside bounding box
    pds = pds[pds_boolean]
    
    if len(pds)!=0:
        pds = pds.reshape((-1,4))

    pedestrian_collided = False 

    car_stop_position = ego_pos
   

    # if ego_speed < STOP_THRESHOLD and len(pds)!=0:
    #    # print("pds",pds)
    #     #print("Pedestrian position",pd_position)
    #     return True, []

    if ego_speed > STOP_THRESHOLD:
        PEDESTRIAN_TRAVELLED_DISTANCE = 0.5 # 0.5 metres pedestrian travelled distance in each frame 
        PEDESTRIAN_SPEED_THRESHOLD = 0.2
        for pd in pds:
            pd_speed = pd[3]
            next_car_center = ego_pos
            if pd_speed > PEDESTRIAN_SPEED_THRESHOLD:
                delta_t = PEDESTRIAN_TRAVELLED_DISTANCE/pd_speed
                ego_travelled_distance = delta_t * ego_speed
                BOUNDING_BOX_LENGTH = 2.3 # approximately half car lenght 
                BOUNDING_BOX_WIDTH = 1.5
                #Computes N_FRAME according lookahead
                n_frames = int(lookahead / ego_travelled_distance)
                
                for i in range(n_frames):
                    # compute new car center according frames already computed and delta t 
                    distance = (i+1)*ego_travelled_distance
                    next_car_center = compute_point_along_direction(ego_pos,ego_yaw,distance)
                    # compute new car bounding_box

                    A,B,C,D = compute_bb_verteces(next_car_center,BOUNDING_BOX_LENGTH,ego_yaw,b=BOUNDING_BOX_WIDTH, b1 = BOUNDING_BOX_LENGTH)
                    bb = Polygon([A,B,C,D,A])

                    pedestrian_bb = pd[0]
                
                    pedestrian_orientation = pd[2]
                    pedestrian_distance = (i+1)*PEDESTRIAN_TRAVELLED_DISTANCE
                    for bb_vertex in pedestrian_bb:
                        new_vertex = compute_point_along_direction(bb_vertex,pedestrian_orientation,pedestrian_distance)
                        point = Point(new_vertex)
                        if bb.contains(point):
                            pedestrian_collided = True
                        break 
                    # stop to simulate collision with current pedestrian
                    if pedestrian_collided:
                        break
            if pedestrian_collided:
                break
        
            dist = np.subtract(car_stop_position,next_car_center)
            norm = np.linalg.norm(dist)
            if norm >=2.3:
                car_stop_position = next_car_center
    



    # # Step 2 compute pedestrian and vehicle trajectory
    # if ego_speed > STOP_THRESHOLD:
    #     # we notice that in general, bounding box vehicle in carla has a x value around 2.3 
    #     distance_along_car_direction = 2.3 # distance from the current car position to next position with fixed orientation
        
    #     # Computes N_FRAME according lookahead
    #     N_FRAMES = int(lookahead / distance_along_car_direction)

    #     delta_t = distance_along_car_direction/ego_speed

        
    #     for i in range(N_FRAMES):
    #         # compute new car center according frames already computed and delta t 
    #         distance = (i+1)*distance_along_car_direction
            
    #         next_car_center = compute_point_along_direction(ego_pos,ego_yaw,distance)
    #         # compute new car bounding_box
    #         A,B,C,D = compute_bb_verteces(next_car_center,distance,ego_yaw,b=1.5,b1=1.5)
    #         bb = Polygon([A,B,C,D,A])
        
    #         # (4,) [list,[x,y],speed,yaw]
    #         for pd in pds:
                
    #             #print("[BP.CHECK_PEDESTRIAN] pd velocity ",pd[3])
    #             distance_along_pedestrian_direction = pd[3] * delta_t*(i+1)
    #             # get bounding box in time t and from this computes new bounding box 
    #             pedestrian_bb = pd[0]
                
    #             pedestrian_orientation = pd[2]
    #             for bb_vertex in pedestrian_bb:
    #                 new_vertex = compute_point_along_direction(bb_vertex,pedestrian_orientation,distance_along_pedestrian_direction)
    #                 point = Point(new_vertex)
    #                 if bb.contains(point):
    #                     pedestrian_collided = True
    #                     break
    #             if pedestrian_collided:
    #                 break
                
    #         if pedestrian_collided:
    #             break
    #         car_stop_position = next_car_center
    

    return flag, car_stop_position


def detect_lead_vehicle(ego_pos,ego_yaw,vehicles,lookahead,looksideways_right=1.5,looksideways_left=1.5):
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
        check_sum = abs(ego_yaw - vehicle.get_orientation()*180/math.pi)
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
            
def compute_separation_distance(ego_speed):
    kmh_speed = ego_speed * 3.6
    separation_distance = math.pow((kmh_speed/10),2)
    return separation_distance
