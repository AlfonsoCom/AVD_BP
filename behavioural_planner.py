#!/usr/bin/env python3
from multiprocessing.connection import wait
import numpy as np
import math
from shapely.geometry import Point, Polygon
from pedestrians import *
from vehicles import *
from traffic_lights import *

# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2

STATES = ["FOLLOW_LANE","DECELERATE_TO_STOP","STAY_STOPPED"]


COUNT_THRESHOLD = 7

# traffic light status
GREEN = 0
YELLOW = 1
RED = 2


BASE_LOOKSIDEWAYS_RIGHT = 3
BASE_LOOKSIDEWAYS_LEFT = 3

MAX_PEDESTRIAN_LOOKSIDEWAYS_LEFT = 5
MAX_PEDESTRIAN_LOOKSIDEWAYS_RIGHT = 4


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
        self._current_traffic_light         = []
        self._pedestrians                   = pedestrians
        self._vehicles                      = vehicles
        self._lead_vehicle                  = None
        self._vehicles_dict                 = None
        self._pedestrian_detected           = False  
        self._car_collision_predicted       = False 
        self._goal_index_to_agent_collision = None



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
        pedestrian_looksideways_left = MAX_PEDESTRIAN_LOOKSIDEWAYS_LEFT
        pedestrian_looksideways_right = MAX_PEDESTRIAN_LOOKSIDEWAYS_RIGHT
        
        
        closest_len, closest_index = get_closest_index(waypoints, ego_state)
        
        if self._state == FOLLOW_LANE:
            # First, find the closest index to the ego vehicle.
            # closest_len, closest_index = get_closest_index(waypoints, ego_state)
            
            # Next, find the goal index that lies within the lookahead distance
            # along the waypoints.
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)


            wp = [ waypoints[goal_index][0], waypoints[goal_index][1],waypoints[goal_index][2]]      

            wp_speed = waypoints[goal_index][2]


            if not self._follow_lead_vehicle:
                # If no lead vehicle is already known search for a lead vehicle to follow

                self._lead_vehicle = detect_lead_vehicle(ego_state[:2],ego_state[2],self._vehicles,self._lookahead )
               
                # if a lead vehicle is detected set true follow_lead_vehicle flag
                if self._lead_vehicle is not None:
                    self._follow_lead_vehicle = True
            else:
                # If a lead vehicle is known checks if it is still a lead vehicle 
                id = self._lead_vehicle.get_id()
                try:
                    self._lead_vehicle = self._vehicles_dict[id]
                    self._lead_vehicle = detect_lead_vehicle(ego_state[:2],ego_state[2],np.array([self._lead_vehicle],dtype=object),self._follow_lead_vehicle_lookahead+10)
                except KeyError:
                    self._lead_vehicle = None
                #self.check_for_lead_vehicle(ego_state, self._lead_vehicle.get_position())
                if self._lead_vehicle is  None:
                    self._follow_lead_vehicle = False


            goal_index_car = goal_index
            goal_index_pd = goal_index
            goal_index_tl = goal_index

            car_collision_predicted, car_stop = check_vehicles(ego_state[:2],ego_state[2],self._vehicles,self._lookahead,
            looksideways_left=3,looksideways_right=3,waypoints=waypoints,closest_index=closest_index,goal_index=goal_index,
            lead_vehicle=self._follow_lead_vehicle)

            self._car_collision_predicted = car_collision_predicted
            if car_collision_predicted:
                if closed_loop_speed > STOP_THRESHOLD:
                    goal_index_car = closest_index
                
                self._goal_index_to_agent_collision = car_stop
                wp_speed = 0
                self._state = DECELERATE_TO_STOP


            ### check pedestrian intersection
            # pedestrian_detected, car_stop = check_pedestrian(ego_state[:2],ego_state[2],closed_loop_speed,
            #     self._pedestrians,lookahead= self._lookahead ,looksideways_right=pedestrian_looksideways_right,looksideways_left=pedestrian_looksideways_left)
            
            
            pedestrian_detected, car_stop = check_pedestrians2(ego_state[:2],ego_state[2],self._pedestrians,
            lookahead= self._lookahead ,looksideways_right=pedestrian_looksideways_right,looksideways_left=pedestrian_looksideways_left,
                waypoints=waypoints,closest_index=closest_index,goal_index=goal_index)

                
            self._pedestrian_detected = pedestrian_detected
            if pedestrian_detected:
                # if closed_loop_speed > STOP_THRESHOLD: # if we detected pedetrian collision the goal index to stop is the closest index 
                goal_index_pd = closest_index

                self._goal_index_to_agent_collision = car_stop

                    #goal_index_pd = get_stop_wp(waypoints,closest_index,goal_index,car_stop)
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
                        self._goal_index_to_agent_collision = goal_index_tl
                        wp_speed = 0
                        self._state = DECELERATE_TO_STOP
            
            goal_index = min(goal_index, goal_index_pd,goal_index_tl)

            wp = [waypoints[goal_index][0],waypoints[goal_index][1],wp_speed]
            self._goal_index = goal_index
            self._goal_state = wp                  

        
        elif self._state == DECELERATE_TO_STOP:

            if abs(closed_loop_speed) <= STOP_THRESHOLD:
                self._state = STAY_STOPPED
                return

            # closest_len, closest_index = get_closest_index(waypoints, ego_state)
            #goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
            goal_index = self._goal_index
            # if new goal_index is greater than last goal_index we don't update the current goal_index
            # because in this state the aim is to decelerate and stop the car.
            # we update the current goal index if the new goal index is smaller or  the latest goal_index
            # is smaller than the actual closest_index. The last condition means that the car passed the latest goal_index and so
            # the new stop goal index will be the closest_index.    
            # if goal_index>self._goal_index:
            #     goal_index = self._goal_index

            if closest_index>self._goal_index:
                closest_index = self._goal_index
            
                
            if closest_index > self._goal_index_to_agent_collision :
                self._goal_index_to_agent_collision = closest_index

            # we chose the goal index to stop according the fact that         
            goal_index_pd = goal_index
            goal_index_tl = goal_index
            goal_index_car = goal_index 

            
            
            car_collision_predicted, car_stop = check_vehicles(ego_state[:2],ego_state[2],self._vehicles,self._lookahead,
            looksideways_left=3,looksideways_right=3,waypoints=waypoints,closest_index=closest_index,goal_index= self._goal_index_to_agent_collision,
            lead_vehicle=False)

            self._car_collision_predicted = car_collision_predicted

            if car_collision_predicted:
                goal_index_car = car_stop


            # pedestrian_detected, car_stop = check_pedestrian(ego_state[:2],ego_state[2],closed_loop_speed,self._pedestrians,lookahead=self._lookahead,looksideways_right=pedestrian_looksideways_right,looksideways_left=pedestrian_looksideways_left)
            pedestrian_detected, car_stop = check_pedestrians2(ego_state[:2],ego_state[2],self._pedestrians,
            lookahead= self._lookahead ,looksideways_right=pedestrian_looksideways_right,looksideways_left=pedestrian_looksideways_left,
                waypoints=waypoints,closest_index=closest_index,goal_index=self._goal_index_to_agent_collision)

            self._pedestrian_detected = pedestrian_detected

            
            
            if pedestrian_detected:
                goal_index_pd = car_stop
                #goal_index_pd = get_stop_wp(waypoints,closest_index,goal_index,car_stop)
                
            traffic_light_on_path = len(self._current_traffic_light)>0
            if traffic_light_on_path:
                id = self._current_traffic_light[0]
                if self._tl_dict[id] != GREEN :
                    tl_position = self._current_traffic_light[1:3]
                    goal_index_tl = get_stop_wp(waypoints,closest_index,goal_index,tl_position)
                elif self._tl_dict[id] == GREEN and not pedestrian_detected and not car_collision_predicted:
                        self._state = FOLLOW_LANE
                        self._current_traffic_light = []
                        self._goal_index_to_agent_collision = None
                        return

            # this condition is when the car has previusly detected a pedestrain on the road
            # and than this pedestrian goes out of road. 

            if not pedestrian_detected and not traffic_light_on_path and not car_collision_predicted:
                    self._state = FOLLOW_LANE
                    self._goal_index_to_agent_collision = None
                    return

            # define goal index according the fact that a pedestrain could be located nearest to the car than
            # the traffic light or viceversa
            goal_index = min(goal_index,goal_index_car,goal_index_pd,goal_index_tl)

            wp = [ waypoints[goal_index][0],waypoints[goal_index][1],0]
            self._goal_index = goal_index
            self._goal_state = wp

        elif self._state == STAY_STOPPED:
            
            # closest_len, closest_index = get_closest_index(waypoints, ego_state)
            if closest_index>self._goal_index:
                closest_index = self._goal_index

            

            car_collision_predicted, _ = check_vehicles(ego_state[:2],ego_state[2],self._vehicles,self._lookahead,
            looksideways_left=3,looksideways_right=3,waypoints=waypoints,closest_index=closest_index,goal_index=self._goal_index)

            self._car_collision_predicted = car_collision_predicted

            

            # cehck if there are some pedetrian along car trajectory
            pedestrian_detected, _ = check_pedestrians2(ego_state[:2],ego_state[2],self._pedestrians,
            lookahead= self._lookahead ,looksideways_right=pedestrian_looksideways_right,looksideways_left=pedestrian_looksideways_left,
                waypoints=waypoints,closest_index=closest_index,goal_index=self._goal_index_to_agent_collision)

            
            self._pedestrian_detected = pedestrian_detected

            # check if there is a red traffic light 
            traffic_light_on_path = len(self._current_traffic_light)>0
            traffic_light_stop = False
            if traffic_light_on_path:
                id = self._current_traffic_light[0]
                traffic_light_stop = self._tl_dict[id] != GREEN

            # if no pedetrain and red traffic light are along car trajectory go to FOLLOW_LANE
            if not pedestrian_detected and not traffic_light_stop and not car_collision_predicted:
                self._state = FOLLOW_LANE
                # reset current traffic light status
                self._goal_index_to_agent_collision
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

