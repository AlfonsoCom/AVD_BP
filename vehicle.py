

class Vehicle():

    def __init__(self,id,position,bounding_box,orientation,speed):
        self._id = id
        self._position = position
        self._bb = bounding_box
        self._orientation = orientation
        self._speed = speed

    def get_id(self):
        return self._id

    def get_position(self):
        pos = self._position
        return [pos.x, pos.y]

    def get_bounding_box(self):
        return self._bb
    
    def get_orientation(self):
        return self._orientation.yaw

    def get_speed(self):
        return self._speed