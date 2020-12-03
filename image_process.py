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
        self.config.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.colorizer = rs.colorizer()

    def on(self):
        self.pipeline.start(self.config)

    def off(self):
        self.pipeline.stop()

    def shotting(self):
        self.capture()
        self.filtering()
        color_image = np.asanyarray(self.color_frame.get_data())

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
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        hole_filling = rs.hole_filling_filter()
        for frame in self.depth_frams:
            frame = depth_to_disparity.process(frame)
            frame = spatial.process(frame)
            frame = temporal.process(frame)
            frame = disparity_to_depth.process(frame)
            frame = hole_filling.process(frame)
        self.aligned_depth_frame = frame.get_data()
        self.colorized_depth = np.asanyarray(
            self.colorizer.colorize(frame).get_data())

    def run(self):
        self.depth_to_world()
        self.solve()

    def solve(self):
        world = self.world.reshape((self.world_x * self.world_y, 3))
        K = np.float32([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
        _, rotation_vector, translation_vector = cv.solvePnP(
            self.world_list, self.image_list, K, np.zeros(5))

        image_points, _ = cv.projectPoints(
            world, rotation_vector, translation_vector, K, np.zeros(5))
        image_points = image_points.reshape((self.world_x, self.world_y, 2))

        self.image_points = image_points
        print(image_points)

    def depth_to_world(self):
        depth_x, depth_y = self.depth.shape
        x = np.linspace(0, depth_x - 1, depth_x)
        y = np.linspace(0, depth_y - 1, depth_y)
        yy, xx = np.meshgrid(x, y)
        self.world = np.zeros((depth_x, depth_y, 3))
        self.world[:, :, 0] = np.swapaxes(yy, 0, 1)
        self.world[:, :, 1] = np.swapaxes(xx, 0, 1)
        self.world[:, :, 2] = self.depth

        self.world_x, self.world_y, _ = self.world.shape
        CENTER_X = int(self.world_x / 2)
        CENTER_C_X = int(CENTER_X / 2)
        CENTER_Y = int(self.world_y / 2)
        CENTER_C_Y = int(CENTER_Y / 2)
        self.world_list = np.array([
            [0, 0, 0],
            [depth_x, 0, 0],
            [0, depth_y, 0],
            [depth_x, depth_y, 0],
            [CENTER_X, CENTER_Y, 0], # center 
            [CENTER_C_X, CENTER_C_Y, 0], # 왼쪽 상단 중간
            [CENTER_X + CENTER_C_X, CENTER_C_Y, 0], # 오른쪽 상단 중간 
            [CENTER_X + CENTER_C_X, CENTER_Y + CENTER_C_Y, 0], # 오른쪽 하단 중간
            [CENTER_C_X, CENTER_Y + CENTER_C_Y, 0], # 왼쪽 하단 중간
        ], dtype=np.float32)

        self.image_list = np.array([
            [-CENTER_X, -CENTER_Y], # 페이지 왼쪽 상단
            [CENTER_X, -CENTER_Y], # 페이지 오른쪽 상단
            [-CENTER_X, CENTER_Y], # 페이지 왼쪽 하단
            [CENTER_X, CENTER_Y], # 페이지 오른쪽 하단
            [0, 0], # 중앙
            [-CENTER_C_X, -CENTER_C_Y], # 왼쪽 상단 중간
            [CENTER_C_X, -CENTER_C_Y], # 오른쪽 상단 중간
            [CENTER_C_X, CENTER_C_Y], # 오른쪽 하단 중간
            [-CENTER_C_X, CENTER_C_Y], # 왼쪽 하단 중간
        ], dtype=np.float32)

    def visual(self):
        # book size
        depth_x, depth_y = self.depth.shape

        # graph axis x,y
        x = np.linspace(0, depth_x - 1, int(depth_x/10))
        y = np.linspace(0, depth_y - 1, int(depth_y/10))
        yy, xx = np.meshgrid(x, y)

        depth = self.depth[::10, ::10].T
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xx, yy, depth, s=1, cmap='Greens')
        plt.show()
