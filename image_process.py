import cv2 as cv
import os
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ImageProcess:
    def __init__(self):
        self.depth_colormap = None
        self.color_image = None
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.colorizer = rs.colorizer()

    def on(self):
        self.pipeline.start(self.config)

    def off(self):
        self.pipeline.stop()

    def shotting(self):
        self.capture()
        self.filtering()
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        color_image = np.asanyarray(self.color_frame.get_data())

        align = rs.align(rs.stream.color)
        frameset = align.process(frames)
        # self.aligned_depth_frame = frameset.get_depth_frame()
        # colorized_depth = np.asanyarray(self.colorizer.colorize(self.aligned_depth_frame).get_data())
        
        self.depth_colormap = np.swapaxes(self.colorized_depth, 0, 1)
        self.color_image = np.swapaxes(color_image, 0, 1)
        self.get_depth_info()

    def get_depth_info(self):
        depth = np.asanyarray(self.aligned_depth_frame)
        self.depth = depth

    def capture(self):
        self.depth_frams = []
        color_frams = []

        for _ in range(10):
            frameset = self.pipeline.wait_for_frames()
            align = rs.align(rs.stream.color)
            frames = align.process(frameset)
            self.depth_frams.append(frames.get_depth_frame())
            color_frams.append(frames.get_color_frame())

        self.color_frame = color_frams[5]
        print('capture!!')

    def filtering(self):
        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)
        decimation = rs.decimation_filter()
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        hole_filling = rs.hole_filling_filter()
        for frame in self.depth_frams:
            # frame = decimation.process(frame)
            frame = depth_to_disparity.process(frame)
            frame = spatial.process(frame)
            frame = temporal.process(frame)
            frame = disparity_to_depth.process(frame)
            frame = hole_filling.process(frame)
        self.aligned_depth_frame = frame.get_data()
        self.colorized_depth = np.asanyarray(self.colorizer.colorize(frame).get_data())

    def run(self):
        pass

    def visual(self):
        # book size
        depth_x, depth_y = self.depth.shape
        
        # graph axis x,y
        x = np.linspace(0, depth_x - 1, int(depth_x/10))
        y = np.linspace(0, depth_y - 1, int(depth_y/10))
        yy, xx = np.meshgrid(x,y)


        depth = self.depth[::10,::10].T
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xx, yy, depth,s=1, cmap='Greens')
        plt.show()



