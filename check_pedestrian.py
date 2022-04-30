import numpy as np, math

PEDESTRIAN_MAX_DISTANCE = 10

# Distance respect to pedestrian at the left while the ego vehicle isn't turning.
PEDESTRIAN_REDUCED_DISTANCE = 5

YAW_OF_TURNING = math.pi / 12

LEFT_RELATIVE_ANGLE = math.pi / 3

#Checks if there is a pedestrian ahead of us.
def check_for_pedestrian(self, ego_state, pedestrian_state):
    """Checks for a specific pedestrian within the proximity 
    of the ego car, such that the ego car should stop.

    args:
        ego_state: ego state vector for the vehicle. (global frame)
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)
        pedestrian_state: state vector for the pedestrian. (global frame)
            format: [ped_x, ped_y, ped_yaw, ped_forward_speed]
                ped_x and ped_y     : position (m)
                ped_yaw             : top-down orientation [-pi to pi]
                ped_forward_speed   : forward speed (m/s)
    returns:
        pedestrian_found: Boolean flag on whether this pedestrian is
            in proximity of the ego car.
    """

    # Check pedestrian position delta vector relative to heading, as well as
    # distance, to determine if there is a pedestrian.
    pedestrian_delta_vector = [(pedestrian_state[0] - ego_state[0]), 
                                (pedestrian_state[1] - ego_state[1])]
    pedestrian_distance = np.linalg.norm(pedestrian_delta_vector)
    
    # Check to see if pedestrian is within range, and is ahead of us.
    # In this case, the pedestrian is too far away.   
    if pedestrian_distance > PEDESTRIAN_MAX_DISTANCE:
        return False

    # Compute the angle between the normalized vector between the pedestrian
    # and ego vehicle position with the ego vehicle's heading vector.
    pedestrian_delta_vector = np.divide(pedestrian_delta_vector, pedestrian_distance)
    ego_heading_vector = [math.cos(ego_state[2]), math.sin(ego_state[2])]

    # Compute the cosine of the angle between ego car and pedestrian respect to Carla
    # cos(a-b) = cos(a)cos(b)+sen(a)sen(b)
    cos = np.dot(pedestrian_delta_vector, ego_heading_vector)

    # Check to see if the relative angle between the pedestrian and the ego
    # vehicle lies within +/- 90 degrees of the ego vehicle's heading 
    # (if cosine of relative angle is negative, it doesn't lies in that range).
    if cos < 0:
        return False
    else:
        return True

    '''
    # Check whether the ego vehicle is turning or not 
    # (if yaw lies within +/- 15 degrees respect the way the car is turning, else not)
    ego_yaw = ego_state[2]
    
    # BASTA QUESTO PER DIRE CHE LA MACCHINA STA SVOLTANDO? PUO' ESSERE CHE LA STRADA NON SIA DRITTA RISPETTO ALL'ORIGINE DEGLI ASSI
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
    '''

if __name__== "__main__":

    # Insert here coordinates to test
    x_v = 8; y_v = 0; yaw_v = math.pi
    x_ped = 10; y_ped = 8
    #################################

    ego_state = [x_v, y_v, yaw_v, 10]
    pedestrian_state = [x_ped, y_ped]

    if check_for_pedestrian(None, ego_state, pedestrian_state):
        print("STOP")
    else:
        print("DON'T STOP")