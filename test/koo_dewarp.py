import os
import sys
import datetime
import cv2
from PIL import Image
import numpy as np
import scipy.optimize

PAGE_MARGIN_X = 50       # reduced px to ignore near L/R edge
PAGE_MARGIN_Y = 20       # reduced px to ignore near T/B edge

OUTPUT_ZOOM = 1.0        # how much to zoom output relative to *original* image
OUTPUT_DPI = 300         # just affects stated DPI of PNG, not appearance
REMAP_DECIMATE = 16      # downscaling factor for remapping image

ADAPTIVE_WINSZ = 55      # window size for adaptive threshold in reduced px

TEXT_MIN_WIDTH = 15      # min reduced px width of detected text contour
TEXT_MIN_HEIGHT = 2      # min reduced px height of detected text contour
TEXT_MIN_ASPECT = 1.5    # filter out text contours below this w/h ratio
TEXT_MAX_THICKNESS = 10  # max reduced px thickness of detected text contour

EDGE_MAX_OVERLAP = 1.0   # max reduced px horiz. overlap of contours in span
EDGE_MAX_LENGTH = 100.0  # max reduced px length of edge connecting contours
EDGE_ANGLE_COST = 10.0   # cost of angles in edges (tradeoff vs. length)
EDGE_MAX_ANGLE = 7.5     # maximum change in angle allowed between contours

RVEC_IDX = slice(0, 3)   # index of rvec in params vector
TVEC_IDX = slice(3, 6)   # index of tvec in params vector
CUBIC_IDX = slice(6, 8)  # index of cubic slopes in params vector

SPAN_MIN_WIDTH = 30      # minimum reduced px width for span
SPAN_PX_PER_STEP = 20    # reduced px spacing for sampling along spans
FOCAL_LENGTH = 1.2       # normalized focal length of camera

DEBUG_LEVEL = 0          # 0=none, 1=some, 2=lots, 3=all
DEBUG_OUTPUT = 'file'    # file, screen, bothpytho

WINDOW_NAME = 'Dewarp'   # Window name for visualization

def resize_to_screen(src,maxw = 1280, maxh = 700):
    height, width = src.shape[:2]
    scl_x = float(width)/maxw
    scl_y = float(height)/maxh
    scl = int(np.ceil(max(scl_x,scl_y)))
    inv_scl = 1 / scl
    img = cv2.resize(src,(0,0),None, inv_scl,inv_scl,cv2.INTER_AREA)
    return img
    
def get_page_extents(rsz_img):
    height, width = rsz_img.shape[:2]

    xmin = PAGE_MARGIN_X
    ymin = PAGE_MARGIN_Y
    xmax = width-PAGE_MARGIN_X
    ymax = height-PAGE_MARGIN_Y

    page = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(page, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)

    outline = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]])

    return page, outline

def box(width, height):
    return np.ones((height, width), dtype=np.uint8)

def get_mask(name, small, pagemask):

    sgray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


    mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV,
                                    ADAPTIVE_WINSZ,
                                    25)

    mask = cv2.dilate(mask, box(9, 1))
    mask = cv2.erode(mask, box(1, 3))

    return np.minimum(mask, pagemask)
def make_tight_mask(contour, xmin, ymin, width, height):

    tight_mask = np.zeros((height, width), dtype=np.uint8)
    tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))
   
    cv2.drawContours(tight_mask, [tight_contour], 0,
                     (1, 1, 1), -1)
    
    return tight_mask
class ContourInfo(object):

    def __init__(self, contour, rect, mask):

        self.contour = contour
        self.rect = rect
        self.mask = mask

        self.center, self.tangent = blob_mean_and_tangent(contour)

        self.angle = np.arctan2(self.tangent[1], self.tangent[0])

        clx = [self.proj_x(point) for point in contour]

        lxmin = min(clx)
        lxmax = max(clx)

        self.local_xrng = (lxmin, lxmax)

        self.point0 = self.center + self.tangent * lxmin
        self.point1 = self.center + self.tangent * lxmax

        self.pred = None
        self.succ = None

    def proj_x(self, point):
        return np.dot(self.tangent, point.flatten()-self.center)

    def local_overlap(self, other):
        xmin = self.proj_x(other.point0)
        xmax = self.proj_x(other.point1)
        return interval_measure_overlap(self.local_xrng, (xmin, xmax))
        
def get_contours(name, small, pagemask):

    mask = get_mask(name, small, pagemask)
    

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)
    
    contours_out = []
    
    for contour in contours:

        rect = cv2.boundingRect(contour)
        
        xmin, ymin, width, height = rect
    
        if (width < TEXT_MIN_WIDTH or
                height < TEXT_MIN_HEIGHT or
                width < TEXT_MIN_ASPECT*height):
            continue

        tight_mask = make_tight_mask(contour, xmin, ymin, width, height)

        if tight_mask.sum(axis=0).max() > TEXT_MAX_THICKNESS:
            continue

        contours_out.append(ContourInfo(contour, rect, tight_mask))

    if DEBUG_LEVEL >= 2:
        visualize_contours(name, small, contours_out)

    return contours_out

def main():
    original_img = cv2.imread("boston_cooking_a.jpg")
    rsz_img = resize_to_screen(original_img)
    # cv2.imshow("rsz_img",rsz_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    pagemask, page_outline = get_page_extents(rsz_img)
    # print(page_outline)

    cinfo_list = get_contours("boston_cooking_a", rsz_img, pagemask)



if __name__ == "__main__":
    main()