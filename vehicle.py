
class Agent():

    def __init__(self,id,position,bounding_box,orientation,speed,type="agents"):
        self._id = id
        self._position = position
        self._bb = bounding_box
        self._orientation = orientation
        self._speed = speed
        self._type = type

    def get_id(self):
        return self._id

    def get_position(self):
        pos = self._position
        return [pos[0], pos[1]]

    def get_bounding_box(self):
        return self._bb
    
    def get_orientation(self):
        return self._orientation

    def get_speed(self):
        return self._speed

    def __str__(self):
        return f"{self._type}(id={self._id}, x={round(self._position.x,2)}, y={round(self._position.y,2)})"





# class Vehicle():

#     def __init__(self,id,position,bounding_box,orientation,speed):
#         self._id = id
#         self._position = position
#         self._bb = bounding_box
#         self._orientation = orientation
#         self._speed = speed

#     def get_id(self):
#         return self._id

#     def get_position(self):
#         pos = self._position
#         return [pos.x, pos.y]

#     def get_bounding_box(self):
#         return self._bb
    
#     def get_orientation(self):
#         return self._orientation

#     def get_speed(self):
#         return self._speed

#     def __str__(self):
#         return f"Vehicle(id={self._id}, x={round(self._position.x,2)}, y={round(self._position.y,2)})"
