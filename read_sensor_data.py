import os, sys
import pickle as pk
import numpy as np, cv2 as cv 
import matplotlib.pyplot as plt

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import live_plotter as lv   # Custom live plotting library
from carla            import sensor
from carla.client     import make_carla_client, VehicleControl
from carla.settings   import CarlaSettings
from carla.tcp        import TCPConnectionError
from carla.controller import utils
from carla.sensor import Camera
from carla.image_converter import labels_to_array, depth_to_array, to_bgra_array, labels_to_cityscapes_palette
from carla.planner.city_track import CityTrack

NONE = 0
PEDESTRIAN = 4
SIDEWALKS = 8
METERS_THRESHOLD = 20.0

def load(filepath):
    with open(filepath, 'rb') as file:
        return pk.load(file)

def show_semSeg_image(obj):
    image_segmentation = labels_to_cityscapes_palette(obj)
    image_segmentation = image_segmentation.astype(np.uint8)
    cv.imshow("Semantic segmentation", image_segmentation)

def show_only_one_class(data, X, Y):
    image = np.zeros((200,200))
    
    for x,y in zip(X,Y):
        image[y,x] = 1

    cv.imshow("Binary image", image)


def show_depth_image(data):
    image_depth = depth_to_array(data)
    cv.imshow("Depth map", image_depth)

def adjacent(a, b):
    '''
    Returns True if a and b are adjacent points
    '''
    if a[0] == b[0] and abs(a[1] - b[1]) == 1:
        return True 
    elif a[1] == b[1] and abs(a[0] - b[0]) == 1:
        return True
    else:
        return False 

'''
Args: 
    semSeg_data: output of the semantic segmentation camera (Image object)
    point: tuple (x,y) 
'''
def point_in_sidewalks(semSeg_data, point):
    
    Y,X = np.where(semSeg_data == SIDEWALKS)
    
    #show_only_one_class(semSeg_data, X,Y)
    #cv.waitKey(0)

    for x,y in zip(X,Y):
        if adjacent((x,y),point):
            return True
    else:
        return False


if __name__ == '__main__':

    semSeg_obj = load("semSeg_data.pkl")
    show_semSeg_image(semSeg_obj)
    semSeg_data = semSeg_obj.data
    #print(semSeg_data.shape) #(200, 200)


    # PEDESTRIAN PIXELS
    Y, X = np.where(semSeg_data == PEDESTRIAN)
    
    #print(X) 
    #print(Y) 
    #coppie di punti il cui pixel corrispondente è occupato da un pedone
    
    '''
    # y
    # |
    # |
    # |
    # |
    # |_______________ x
    

    depth_data = load("depth_data.pkl")
    show_depth_image(depth_data)
    depth_data = depth_data.data
    #print(depth_data.shape) #(200, 200)
    #print(depth_data) #normalized in range(0,1) -> 1 is the farest, 0 is the nearest
    
    triplets = []
    newX = []
    newY = []
    for x,y in zip(X,Y):
        z = depth_data[y,x] * 1000
        if z < METERS_THRESHOLD:
            t = (x,y,z)
            #print(t)
            triplets.append(t)
            newX.append(x)
            newY.append(y)

    plt.scatter(newX,newY, s=1)
    plt.xlim(0,200)
    plt.ylim(0,200)
    plt.show()'''

    new_image = np.zeros((200,200))
    newX = [41,63,122]
    newY = [108,106,105]

    for x,y in zip(newX, newY):
        new_image[y,x]=1
        if point_in_sidewalks(semSeg_data, (x,y)):
            print(f"{(x,y)} lies in sidewalks")
        else:
            print(f"{(x,y)} doesn't lie in sidewalks")
    
    cv.imshow("Punti di interesse", new_image)

    cv.waitKey(0)