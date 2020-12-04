from tkinter.constants import CENTER
import cv2 as cv
import os
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ImageProcess:
    def __init__(self):
        self.IMAGE_SIZE = (1280, 720)
        self.depth_colormap = None
        self.color_image = None
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], rs.format.z16, 6)
        self.config.enable_stream(
            rs.stream.color, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], rs.format.bgr8, 6)
        self.colorizer = rs.colorizer()

    def on(self):
        self.pipeline.start(self.config)

    def off(self):
        self.pipeline.stop()

    def shotting(self):
        self.capture()
        self.filtering()
        self.color_image_orign = np.asanyarray(self.color_frame.get_data())

        self.depth_colormap = np.swapaxes(self.colorized_depth, 0, 1)
        self.color_image = np.swapaxes(self.color_image_orign, 0, 1)
        self.get_depth_info()

    def get_depth_info(self):
        depth = np.asanyarray(self.aligned_depth_frame)
        depth_zero = depth[np.where(depth >= 200)].min()
        depth = depth - depth_zero
        self.interpolation(depth)

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
        self.remap()

    def interpolation(self, depth):
        depth_scale = depth[::100, ::100]
        self.depth = cv.resize(depth_scale, dsize=self.IMAGE_SIZE, interpolation=cv.INTER_CUBIC)

    def remap(self):
        img_gray = cv.cvtColor(self.color_image_orign, cv.COLOR_BGR2GRAY)
        image_height_coords = self.image_points[:, 0, 0].reshape(self.depth_height_coords.shape).astype(np.float32)
        image_width_coords = self.image_points[:, 0, 1].reshape(self.depth_width_coords.shape).astype(np.float32)

        remapped = cv.remap(img_gray, image_height_coords, image_width_coords, cv.INTER_CUBIC, None, cv.BORDER_REPLICATE)
        plt.imshow(remapped); plt.show()

    def solve(self):
        K = np.float32([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
        _, rotation_vector, translation_vector = cv.solvePnP(
            self.world_list, self.image_list, K, np.zeros(5))

        image_points, _ = cv.projectPoints(
            self.world, rotation_vector, translation_vector, K, np.zeros(5))
        
        self.image_points = image_points

    def depth_to_world(self):
        depth_height, depth_width = self.depth.shape
        depth_height_lin = np.linspace(0, depth_height - 1, depth_height)
        depth_width_lin = np.linspace(0, depth_width - 1, depth_width)
        self.depth_height_coords, self.depth_width_coords = np.meshgrid(
            depth_height_lin, depth_width_lin)
        depth_hw_coords = np.hstack((self.depth_height_coords.flatten().reshape((-1, 1)),
                                     self.depth_width_coords.flatten().reshape((-1, 1))))
        depth_coords = self.depth.T.flatten()
        self.world = np.hstack((depth_hw_coords, depth_coords.reshape((-1,1))))
        
        CENTER_WIDTH = int(depth_width / 2)
        CENTER_C_WIDTH = int(CENTER_WIDTH / 2)
        CENTER_HEIGHT = int(depth_height / 2)
        CENTER_C_HEIGHT = int(CENTER_HEIGHT / 2)
        self.world_list = np.array([
            [0, 0, 0], # page 왼쪽 상단
            [depth_height, 0, 0], # page 왼쪽 아래
            [0, depth_width, 0], # page 오른쪽 상단
            [depth_height, depth_width, 0], # page 오른쪽 하단
            [CENTER_HEIGHT, CENTER_WIDTH, 0],  # center
            [CENTER_C_HEIGHT, CENTER_C_WIDTH, 0],  # 왼쪽 상단 중간
            [CENTER_C_HEIGHT, CENTER_WIDTH + CENTER_C_WIDTH , 0],  # 오른쪽 상단 중간
            [CENTER_HEIGHT + CENTER_C_HEIGHT, CENTER_WIDTH + CENTER_C_WIDTH, 0],  # 오른쪽 하단 중간
            [CENTER_HEIGHT + CENTER_C_HEIGHT ,CENTER_C_WIDTH, 0],  # 왼쪽 하단 중간
        ], dtype=np.float32)

        self.image_list = np.array([
            [0, 0], # page 왼쪽 상단
            [depth_height, 0], # page 왼쪽 아래
            [0, depth_width], # page 오른쪽 상단
            [depth_height, depth_width], # page 오른쪽 하단
            [CENTER_HEIGHT, CENTER_WIDTH],  # center
            [CENTER_C_HEIGHT, CENTER_C_WIDTH],  # 왼쪽 상단 중간
            [CENTER_C_HEIGHT, CENTER_WIDTH + CENTER_C_WIDTH ],  # 오른쪽 상단 중간
            [CENTER_HEIGHT + CENTER_C_HEIGHT, CENTER_WIDTH + CENTER_C_WIDTH],  # 오른쪽 하단 중간
            [CENTER_HEIGHT + CENTER_C_HEIGHT ,CENTER_C_WIDTH]  # 왼쪽 하단 중간
        ], dtype=np.float32)

    def visual(self):
        # book size
        depth_x, depth_y = self.depth.shape

        # graph axis x,y
        x = np.linspace(0, depth_x - 1, int(depth_x/10))
        y = np.linspace(0, depth_y - 1, int(depth_y/10))
        yy, xx = np.meshgrid(x, y)

        depth_sclae = self.depth[::10, ::10].T
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xx, yy, depth_sclae, s=1, cmap='Greens')
        plt.show()
