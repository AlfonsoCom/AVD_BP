import numpy as np
import math
from utils import *
from shapely.geometry import Point, Polygon, LineString

# Stop speed threshold
STOP_THRESHOLD = 0.3
# value to add to CAR ROI 
CAR_LATERAL_MARGIN = 0.5

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
        
            # (4,) [list,[x,y],yaw,speed]
            for pd in pds:
                
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
    

    return flag and pedestrian_collided, car_stop_position


def check_pedestrians2(ego_pos,ego_yaw,pedestrians,lookahead,looksideways_right,looksideways_left,waypoints,closest_index,goal_index):
    """
        Predict the presence of a pedestrian on the ego trajectory. If a pedestrian collision is estimated returns 
        true flag and the new goal index to stop.
        params:
            ego_pos list([x,y]):  car coordinates
            ego_yaw float : car orientation
            pedestrians np.array(): array of vehicles
            lookahead float:  max  look a head distance 
            looksideways_right float: max look right distance 
            looksideways_left float: max looksideways left distance
            waypoints np.array(): point that car should follow
            closest_index int:  the nearest waypoint index
            goal_index int: the goal waypoint index
    """
    if len(pedestrians)==0:
        return False,goal_index

    # Step 1 filter pedetrians in bb
    A,B,C,D = compute_bb_verteces(ego_pos,lookahead,ego_yaw,looksideways_right,looksideways_left)
    bb = Polygon([A,B,C,D,A])
    
    # numpy array pedestrians
    pds  = pedestrians
    pds_boolean = pds == None
    for i,pd in enumerate(pds):
        # get pedestrian bb
        pedestrian_bb_verteces = pd.get_bounding_box()
        for bb_vertex in pedestrian_bb_verteces:
            
            vertex = Point(bb_vertex)
            if bb.contains(vertex):
                pds_boolean[i] = True
                break
        
      
    # considered only pedestrians inside bounding box
    pds = pds[pds_boolean]
    
    start_point = ego_pos
    intersected = False
    index = closest_index
    # plus one so in this way also goal index is used to check intersection
    for index in range(closest_index,goal_index+1):
        next_point = waypoints[index][:2]
        v_diff = np.subtract(next_point,start_point)
        norm = np.linalg.norm(v_diff)
        orientation = math.atan2(v_diff[1],v_diff[0])
        car_extent_y = 1.5
        A,B,C,D = compute_bb_verteces(start_point,norm,orientation,car_extent_y+CAR_LATERAL_MARGIN)
        car_path = Polygon([A,B,C,D,A])
        #car_path = LineString([start_point,next_point])
        pd_distance = 15 # in further work udapte this
        for pd in pds:
            pd_start_point = pd.get_position()
            pd_next_point = compute_point_along_direction(pd_start_point,pd.get_orientation(),pd_distance)
            pd_path = LineString([pd_start_point,pd_next_point])
            intersected = pd_path.intersects(car_path)
            if intersected:
                return intersected,index
        start_point = next_point

    return intersected,index
        


        