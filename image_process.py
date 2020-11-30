import cv2 as cv
import os
import numpy as np
import pyrealsense2 as rs

class ImageProcess:
    def __init__(self):
        self.depth_colormap = None
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        

    def shotting(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
        self.depth_colormap = depth_colormap
