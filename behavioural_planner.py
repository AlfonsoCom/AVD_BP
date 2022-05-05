#!/usr/bin/env python3
import numpy as np
import math
from shapely.geometry import Point, Polygon

# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2

STATES = ["FOLLOW_LANE","DECELERATE_TO_STOP","STAY_STOPPED"]
# Stop speed threshold
STOP_THRESHOLD = 0.05
# Number of cycles before moving from stop sign.
STOP_COUNTS = 10

# traffic light status
GREEN = 0
YELLOW = 1
RED = 2

# semicircle radius used to detect traffic lights
RADIUS = 50 # metres

class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead,traffic_lights,tl_dict,pedestrians=[]):
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
        self._tl_id                         = None
        self._pedestrians                   = pedestrians

    def set_lookahead(self, lookahead):
        self._lookahead = lookahead

    def set_tl_dict(self,tl_dict):
        self._tl_dict = tl_dict

    def set_pedestrians(self,pedestrians):
        self._pedestrians  = pedestrians

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
        print("\n[BP.TRANSITION_STATE] CURRENT STATE -> " ,STATES[self._state])
        if self._state == FOLLOW_LANE:
            # First, find the closest index to the ego vehicle.
            closest_len, closest_index = get_closest_index(waypoints, ego_state)
            
            # Next, find the goal index that lies within the lookahead distance
            # along the waypoints.
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)

           
           

            ### check pedestrian intersection
            collisioned, car_stop = check_pedestrian(ego_state[:2],ego_state[2],closed_loop_speed,
                self._pedestrians,self._lookahead,looksideways_right=3,looksideways_left=4)
            
            wp = [ waypoints[goal_index][0], waypoints[goal_index][1],waypoints[goal_index][2]]

            if collisioned:
                if closed_loop_speed > STOP_THRESHOLD:
                    goal_index = get_stop_wp(waypoints,closest_index,goal_index,car_stop)
                wp = [ waypoints[goal_index][0],waypoints[goal_index][1],0] 
                self._state = DECELERATE_TO_STOP
            else:            
                tl = check_traffic_light(ego_state[:2],ego_state[2],self._traffic_lights,self._lookahead,looksideways_right=3.5)
                status = None
                if len(tl)>0:
                    status = self._tl_dict[tl[0]]             
                    if status != GREEN: 
                        self._tl_id = tl[0]
                        goal_index = get_stop_wp(waypoints,closest_index,goal_index,tl[1:3])
                        wp = [ waypoints[goal_index][0],waypoints[goal_index][1],0]
                        self._state = DECELERATE_TO_STOP




            self._goal_index = goal_index
            self._goal_state = wp
            

            
            

        # In this state, check if we have reached a complete stop. Use the
        # closed loop speed to do so, to ensure we are actually at a complete
        # stop, and compare to STOP_THRESHOLD.  If so, transition to the next
        # state.
        elif self._state == DECELERATE_TO_STOP:

            if abs(closed_loop_speed) <= STOP_THRESHOLD:
                self._state = STAY_STOPPED
                return

            closest_len, closest_index = get_closest_index(waypoints, ego_state)
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
            collisioned, car_stop = check_pedestrian(ego_state[:2],ego_state[2],closed_loop_speed,self._pedestrians,self._lookahead,looksideways_right=3,looksideways_left=4)
            wp = [ waypoints[goal_index][0],waypoints[goal_index][1],waypoints[goal_index][2]]
            if collisioned:
                goal_index = get_stop_wp(waypoints,closest_index,goal_index,car_stop)
                wp = [ waypoints[goal_index][0],waypoints[goal_index][1],0]
                self._goal_index = goal_index
                self._goal_state = wp
            else:
                try:
                    if self._tl_dict[self._tl_id] == GREEN:
                        self._state = FOLLOW_LANE
                        self._tl_id = None
                except KeyError:
                    # if no traffic light are previusly detected so 
                    # the next state will be follow lane because there are
                    # neither pedestrians and traffic light
                    self._state = FOLLOW_LANE
                    # means that no traffic light is chcecked
            

        # In this state, check to see if we have stayed stopped for at
        # least STOP_COUNTS number of cycles. If so, we can now leave
        # the stop sign and transition to the next state.
        elif self._state == STAY_STOPPED:
            check_pedestrian_collision, car_stop = check_pedestrian(ego_state[:2],ego_state[2],closed_loop_speed,self._pedestrians,lookahead=self._lookahead,looksideways_right=3,looksideways_left=4)
            print("[BP.TRANSITION_STATE.STAY_STOPPED] Check pedestrian",check_pedestrian_collision)

            traffic_light_condition = True if self._tl_id is None else self._tl_dict[self._tl_id] == GREEN
            if not check_pedestrian_collision and traffic_light_condition:
                self._state = FOLLOW_LANE
                # if status equal to green
                self._tl_id = None
           
            
           
        else:
            raise ValueError('Invalid state value.')
        
        print("[BP.transistion_state] closest_wp_index - goal_wp_index ", closest_index,self._goal_index )

  
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

            # Add a 15m buffer to prevent oscillations for the distance check.
            if lead_car_distance < self._follow_lead_vehicle_lookahead + 15:
                return
            # Check to see if the lead vehicle is still within the ego vehicle's
            # frame of view.
            lead_car_delta_vector = np.divide(lead_car_delta_vector, lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), math.sin(ego_state[2])]
            if np.dot(lead_car_delta_vector, ego_heading_vector) > (1 / math.sqrt(2)):
                return

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

def get_stop_wp(waypoints, closest_index,goal_index,traffic_light_pos):
    # note -> this function works only if goal_index - closest_index > 2
    for i in range(closest_index,goal_index):
        dist_wps = np.subtract(waypoints[i+1][:2],waypoints[i][:2])
        s2 = np.add(traffic_light_pos,[dist_wps[1],dist_wps[0]])
        reference_vector = np.subtract(s2,traffic_light_pos)
        v1 = np.subtract(waypoints[i][:2],traffic_light_pos)
        v2 = np.subtract(waypoints[i+1][:2],traffic_light_pos)
        sign_1 = np.sign(np.cross(reference_vector,v1))
        sign_2 = np.sign(np.cross(reference_vector,v2))

        
        if (sign_1 == 0) and pointOnSegment(traffic_light_pos, waypoints[i][:2], s2):
            return i-1
        if (sign_2 == 0) and pointOnSegment(traffic_light_pos, waypoints[i+1][:2], s2):
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
    
    # center_point = Point(ego_pos)
        
    A,B,C,D = compute_bb_verteces(ego_pos,lookahead,ego_yaw,looksideways_right,looksideways_left)
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

    THRESHOLD_DEGREE = 2.5
    
    ego_yaw = ego_yaw*180/math.pi
    # 180 - ( traffic_lights_yaw + car_yaw)
    #index_tl_opposite = np.where( np.abs(REFERENCE_ANGLE-(np.abs(tl[:,3])+abs(ego_yaw))) <= THRESHOLD_DEGREE)[0]
    check_sum_90 = np.abs(90-(tl[:,3]+abs(ego_yaw)))<=THRESHOLD_DEGREE
    check_sum_270 = np.abs(270-(tl[:,3]+abs(ego_yaw)))<=THRESHOLD_DEGREE
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
    
    print("[BP.CHECK_PEDESTRIANS] ego (x,y,teta): ", ego_pos,ego_yaw)
    # numpy array pedestrians
    pds  = pedestrians
    pds_boolean = pds == None
    pd_position  = []
    pd_speed = None
    pd_orientation = None
    flag = False
    for i,pd in enumerate(pds):
        # get pedestrian bb
        pedestrian_bb_verteces = pd[0]
        

        for bb_vertex in pedestrian_bb_verteces:
            
            vertex = Point(bb_vertex)
            if bb.contains(vertex):
                flag = True
                pds_boolean[i] = True
                pd_position = pd[1]
                pd_speed = pd[-1]
                pd_orientation = pd[2]
                break
        
      
    # considered only pedestrians inside bounding box
    pds = pds[pds_boolean]
 
    print("[BP.CHECK_PEDESTRIAN] pd_collided_pedestrian_position ->",pd_position)
    print("[BP.CHECK_PEDESTRIAN] pd_collided_pedestrian_speed ->",pd_speed)
    print("[BP.CHECK_PEDESTRIAN] pd_collided_pedestrian_orientation ->",pd_orientation)


    
    if len(pds)!=0:
        pds = pds.reshape((-1,4))

    print("[BP.CHECK_PEDESTRIAN] pds.shape ->",pds.shape)

    pedestrian_collided = False 

    car_stop_position = ego_pos

    if ego_speed < STOP_THRESHOLD and len(pds)!=0:
       # print("pds",pds)
        #print("Pedestrian position",pd_position)
        return True, []


    # Step 2 compute pedestrian and vehicle trajectory
    if ego_speed > STOP_THRESHOLD:
        # we notice that in general, bounding box vehicle in carla has a x value around 2.3 
        distance_along_car_direction = 2.3 # distance from the current car position to next position with fixed orientation
        
        # Computes N_FRAME according lookahead
        N_FRAMES = int(lookahead / distance_along_car_direction)

        delta_t = distance_along_car_direction/ego_speed

        
        for i in range(N_FRAMES):
            # compute new car center according frames already computed and delta t 
            distance = (i+1)*distance_along_car_direction
            
            next_car_center = compute_point_along_direction(ego_pos,ego_yaw,distance)
            # compute new car bounding_box
            A,B,C,D = compute_bb_verteces(next_car_center,distance,ego_yaw,b=1.5,b1=1.5)
            bb = Polygon([A,B,C,D,A])
        
            # (4,) [list,[x,y],speed,yaw]
            for pd in pds:
                
                #print("[BP.CHECK_PEDESTRIAN] pd velocity ",pd[3])
                distance_along_pedestrian_direction = pd[3] * delta_t*(i+1)
                # get bounding box in time t and from this computes new bounding box 
                pedestrian_bb = pd[0]
                
                pedestrian_orientation = pd[2]
                for bb_vertex in pedestrian_bb:
                    new_vertex = compute_point_along_direction(bb_vertex,pedestrian_orientation,distance_along_pedestrian_direction)
                    point = Point(new_vertex)
                    if bb.contains(point):
                        pedestrian_collided = True
                        break
                if pedestrian_collided:
                    break
                
            if pedestrian_collided:
                break
            car_stop_position = next_car_center
    

    return flag, car_stop_position
