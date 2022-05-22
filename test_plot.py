import numpy as np
from math import cos,sin 
def to_rot(r):
    Rx = np.mat([[ 1,         0,           0],
                 [ 0, cos(r[0]), -sin(r[0]) ],
                 [ 0, sin(r[0]),  cos(r[0]) ]])

    Ry = np.mat([[ cos(r[1]), 0,  sin(r[1]) ],
                 [ 0,         1,          0 ],
                 [-sin(r[1]), 0,  cos(r[1]) ]])

    Rz = np.mat([[ cos(r[2]), -sin(r[2]), 0 ],
                 [ sin(r[2]),  cos(r[2]), 0 ],
                 [         0,          0, 1 ]])

    return Rz*Ry*Rx

def rotate_x(angle):
    R = np.mat([[ 1,         0,           0],
                 [ 0, cos(angle), -sin(angle) ],
                 [ 0, sin(angle),  cos(angle) ]])
    return R

def rotate_y(angle):
    R = np.mat([[ cos(angle), 0,  sin(angle) ],
                 [ 0,         1,          0 ],
                 [-sin(angle), 0,  cos(angle) ]])
    return R

def rotate_z(angle):
    R = np.mat([[ cos(angle), -sin(angle), 0 ],
                 [ sin(angle),  cos(angle), 0 ],
                 [         0,          0, 1 ]])
    return R


r_t = np.dot(rotate_z(-90*np.pi/180),rotate_x(-90*np.pi/180))
print(np.round(r_t,2))