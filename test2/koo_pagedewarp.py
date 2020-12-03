import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from cv2 import cv2
import scipy.integrate 


a = 0.0001
b = -0.02
c = 1
d = 0


def main():
    # img = cv2.imread("view.jpg", cv2.IMREAD_GRAYSCALE)
    # dst = cv2.resize(img, dsize=(201,101), interpolation=cv2.INTER_AREA)
    book, booklist, imagelist = return_book_index()
    image_points = solve(book,booklist,imagelist)
    print(image_points[100,50])
    


def return_book_index():
    x = np.linspace(0,100,101)
    y = np.linspace(0,200,201)
    yy, xx = np.meshgrid(x,y)
    z = a * yy**3 + b*yy**2 + c*yy + d
    world = np.array([list(zip(x,y,z)) for x,y,z in zip(xx,yy,z)])
    y = f(x)
    len100 = int(integrate(f,100)[0])
    len0 = int(integrate(f,0)[0])
    
    worldlist = [world[200,0],world[200,100],world[0,0],world[0,100]]
    imagelist = [np.array([200,len0]), np.array([200,len100]), np.array([0,len0]), np.array([0,len100])]
    worldlist = np.array(worldlist, dtype=np.float32)
    imagelist = np.array(imagelist, dtype=np.float32)
    return world, worldlist, imagelist

def f(x):
    return (1 + (3*a * x**2 + 2*b*x + c)**2)**(1/2)

def integrate(fx,x):
    return scipy.integrate.quad(fx,0,x)

def solve(world,worldlist,imagelist):
    worldx = world.shape[0]
    worldy = world.shape[1]
    world = world.reshape((worldx*worldy,3))
    K = np.float32([[1,0,0],
                [0,1,0],
                [0,0,1]])
    _, rotation_vector, translation_vector = cv2.solvePnP(worldlist, imagelist, K, np.zeros(5))

    image_points, _ = cv2.projectPoints(world,rotation_vector,translation_vector,K, np.zeros(5))
    image_points = image_points.reshape((worldx,worldy,2))
    
    return image_points






if __name__ == "__main__":
    main()
