#!/usr/bin/env python3
from multiprocessing.connection import wait
import numpy as np
import math
from shapely.geometry import Point, Polygon
from pedestrians import *
from vehicles import *
from traffic_lights import *
from utils import get_closest_index
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

PEDESTRIAN_LOOKSIDEWAYS_LEFT = 5
PEDESTRIAN_LOOKSIDEWAYS_RIGHT = 4


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
        # pedestrian_looksideways_left = MAX_PEDESTRIAN_LOOKSIDEWAYS_LEFT
        # pedestrian_looksideways_right = MAX_PEDESTRIAN_LOOKSIDEWAYS_RIGHT
        
        
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
                if self._lead_vehicle is  None:
                    self._follow_lead_vehicle = False


            goal_index_car = goal_index
            goal_index_pd = goal_index
            goal_index_tl = goal_index

            # check car collisions
            car_collision_predicted, car_stop = check_vehicles(ego_state[:2],ego_state[2],self._vehicles,self._lookahead,
            looksideways_left=BASE_LOOKSIDEWAYS_LEFT,looksideways_right=BASE_LOOKSIDEWAYS_RIGHT,waypoints=waypoints,closest_index=closest_index,goal_index=goal_index,
            lead_vehicle=self._follow_lead_vehicle)

            self._car_collision_predicted = car_collision_predicted
            if car_collision_predicted:
                if closed_loop_speed > STOP_THRESHOLD:
                    goal_index_car = closest_index
                
                self._goal_index_to_agent_collision = car_stop
                wp_speed = 0
                self._state = DECELERATE_TO_STOP


            # check pedestrian collisions
            
            pedestrian_detected, car_stop = check_pedestrians(ego_state[:2],ego_state[2],self._pedestrians,
            lookahead= self._lookahead ,looksideways_right=PEDESTRIAN_LOOKSIDEWAYS_RIGHT,looksideways_left=PEDESTRIAN_LOOKSIDEWAYS_LEFT,
                waypoints=waypoints,closest_index=closest_index,goal_index=goal_index)

                
            self._pedestrian_detected = pedestrian_detected
            if pedestrian_detected:
                # if closed_loop_speed > STOP_THRESHOLD: # if we detected pedetrian collision the goal index to stop is the closest index 
                goal_index_pd = closest_index

                self._goal_index_to_agent_collision = car_stop

                    #goal_index_pd = get_stop_wp(waypoints,closest_index,goal_index,car_stop)
                wp_speed = 0
                self._state = DECELERATE_TO_STOP
                       
            # check red traffic lights
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

            # maintain consistency with the previously estimated collision
            goal_index = self._goal_index

            if closest_index>self._goal_index:
                closest_index = self._goal_index
            
                
            if closest_index > self._goal_index_to_agent_collision :
                self._goal_index_to_agent_collision = closest_index

            # we chose the goal index to stop according the fact that         
            goal_index_pd = goal_index
            goal_index_tl = goal_index
            goal_index_car = goal_index 

            
            # check if there are some vehicles along car trajectory
            car_collision_predicted, car_stop = check_vehicles(ego_state[:2],ego_state[2],self._vehicles,self._lookahead,
            looksideways_left=BASE_LOOKSIDEWAYS_LEFT,looksideways_right=BASE_LOOKSIDEWAYS_RIGHT,waypoints=waypoints,closest_index=closest_index,goal_index= self._goal_index_to_agent_collision,
            lead_vehicle=False)

            self._car_collision_predicted = car_collision_predicted

            if car_collision_predicted:
                goal_index_car = car_stop

            # check if there are some pedetrian along car trajectory
            pedestrian_detected, car_stop = check_pedestrians(ego_state[:2],ego_state[2],self._pedestrians,
            lookahead= self._lookahead ,looksideways_right=PEDESTRIAN_LOOKSIDEWAYS_RIGHT,looksideways_left=PEDESTRIAN_LOOKSIDEWAYS_LEFT,
                waypoints=waypoints,closest_index=closest_index,goal_index=self._goal_index_to_agent_collision)

            self._pedestrian_detected = pedestrian_detected
            
            if pedestrian_detected:
                goal_index_pd = car_stop
                
            # check if there are a red traffic light trian along car trajectory
            
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
            
            # to maintain consistency with the previously estimated collision
            if closest_index>self._goal_index:
                closest_index = self._goal_index

            # check if there are some vehicles along car trajectory
            car_collision_predicted, _ = check_vehicles(ego_state[:2],ego_state[2],self._vehicles,self._lookahead,
            looksideways_left=3,looksideways_right=3,waypoints=waypoints,closest_index=closest_index,goal_index=self._goal_index)

            self._car_collision_predicted = car_collision_predicted

            

            # check if there are some pedetrian along car trajectory
            pedestrian_detected, _ = check_pedestrians(ego_state[:2],ego_state[2],self._pedestrians,
            lookahead= self._lookahead ,looksideways_right=BASE_LOOKSIDEWAYS_LEFT,looksideways_left=BASE_LOOKSIDEWAYS_RIGHT,
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
                
 
