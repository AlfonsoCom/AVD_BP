import matplotlib.pyplot as plt
import numpy as np
points = [[321.2112439926337, 206.50232399398996], [321.4720334853697, 206.7357524333588], [321.7328229781057, 206.96918087272766], [322.0329452544548, 206.6338800995251], [322.3330675308038, 206.29857932632254], [322.0722780380678, 206.0651508869537], [321.8114885453318, 205.83172244758484], [321.5113662689827, 206.1670232207874]]  
#takes only verteces of pedestrains bb
points = points[0:-1:2]

points = np.array(points)
# points = points[np.array([x for x in range(0,len(points),2)])]
for i,point in enumerate(points):
    plt.scatter(point[0],point[1],label=str(i))
plt.legend()
plt.show()