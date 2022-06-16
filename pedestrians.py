import numpy as np
import math
from utils import *
from shapely.geometry import Point, Polygon, LineString

# Stop speed threshold
STOP_THRESHOLD = 0.3
# value to add to CAR ROI 
CAR_LATERAL_MARGIN = 0.5



def check_pedestrians(ego_pos,ego_yaw,pedestrians,lookahead,looksideways_right,looksideways_left,waypoints,closest_index,goal_index,car_extent=1.5,projection_length=15):
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
            car_extent float: area width between two consecutive waypoints 
            projection_length float: vehicle projection length 
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
        A,B,C,D = compute_bb_verteces(start_point,norm,orientation,car_extent+CAR_LATERAL_MARGIN)
        car_path = Polygon([A,B,C,D,A])
        for pd in pds:
            pd_start_point = pd.get_position()
            pd_next_point = compute_point_along_direction(pd_start_point,pd.get_orientation(),projection_length)
            pd_path = LineString([pd_start_point,pd_next_point])
            intersected = pd_path.intersects(car_path)
            if intersected:
                return intersected,index
        start_point = next_point

    return intersected,index
        


        