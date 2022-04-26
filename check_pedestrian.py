import numpy as np, math

PEDESTRIAN_MAX_DISTANCE = 5

# Distance respect to pedestrian at the left while the ego vehicle isn't turning.
PEDESTRIAN_REDUCED_DISTANCE = 3

YAW_OF_TURNING = math.pi / 12

LEFT_RELATIVE_ANGLE = math.pi / 3

#Checks if there is a pedestrian ahead of us.
def check_for_lead_vehicle(self, ego_state, pedestrian_position):
    """Checks for lead vehicle within the proximity of the ego car, such
    that the ego car should begin to follow the lead vehicle.

    args:
        ego_state: ego state vector for the vehicle. (global frame)
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)
        pedestrian_position: The [x, y] position of the pedestrian.
            Lengths are in meters, and it is in the global frame.
    returns:
        pedestrian_checked: Boolean flag on whether there is a pedestrian 
            in proximity of the ego car.
    """

    # Check pedestrian position delta vector relative to heading, as well as
    # distance, to determine if there is a pedestrian.
    pedestrian_delta_vector = [pedestrian_position[0] - ego_state[0], 
                                pedestrian_position[1] - ego_state[1]]
    pedestrian_distance = np.linalg.norm(pedestrian_delta_vector)
    
    # Check to see if pedestrian is within range, and is ahead of us.
    # In this case, the pedestrian is too far away.   
    if pedestrian_distance > PEDESTRIAN_MAX_DISTANCE:
        return False

    # Compute the angle between the normalized vector between the pedestrian
    # and ego vehicle position with the ego vehicle's heading vector.
    pedestrian_delta_vector = np.divide(pedestrian_delta_vector, 
                                        pedestrian_distance)
    ego_heading_vector = [math.cos(ego_state[2]), 
                            math.sin(ego_state[2])]

    # Check to see if the relative angle between the pedestrian and the ego
    # vehicle lies within +/- 90 degrees of the ego vehicle's heading 
    # (if cosine of relative angle is negative, it doesn't lies in that range).
    if pedestrian_delta_vector[0] < 0:
        return False

    # Check whether the ego vehicle is turning or not 
    # (if yaw lies within +/- 15 degrees respect the way the car is turning, else not)
    ego_yaw = ego_state[2]
    
    #conditions = [ego_yaw >= x - YAW_OF_TURNING and ego_yaw <= x + YAW_OF_TURNING for x in (0, math.pi/2, math.pi, -math.pi/2)]
    zero_condition = ego_yaw >= 0 - YAW_OF_TURNING and ego_yaw <= 0 + YAW_OF_TURNING
    pi2_condition = ego_yaw >= math.pi/2 - YAW_OF_TURNING and ego_yaw <= math.pi/2 + YAW_OF_TURNING
    pi_condition = ego_yaw >= math.pi - YAW_OF_TURNING and ego_yaw <= math.pi + YAW_OF_TURNING
    minus_pi2_condition = ego_yaw >= - math.pi/2 - YAW_OF_TURNING and ego_yaw <= - math.pi/2 + YAW_OF_TURNING

    if not (zero_condition or pi2_condition or pi_condition or minus_pi2_condition):
        # If the car is turning, the pedestrian is considered in proximity of the car 
        return True
    else:
        # If the car isn't turning, the distance at with consider the pedestrian at the left is reduced

        # If the pedestrian is at the left respect to the car and it's at enough distance, the car will not consider it
        if pedestrian_delta_vector[1] > math.sin(LEFT_RELATIVE_ANGLE) and pedestrian_distance > PEDESTRIAN_REDUCED_DISTANCE:
            return False
        else:
            return True

