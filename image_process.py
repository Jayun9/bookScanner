from cv2 import cv2 as cv
import os
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import scipy.optimize
import crop
import page_dewarp


class ImageProcess:
    def __init__(self):
        self.img_number = 1
        self.iscaptured = False
        self.IMAGE_SIZE = (1280, 720)
        self.depth_colormap = None
        self.color_image = None
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.depth, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], rs.format.z16, 6)
        self.config.enable_stream(
            rs.stream.color, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], rs.format.bgr8, 6)
        self.colorizer = rs.colorizer()
        self.input_image = None
        self.stream_stop = False

    def save(self, svae_folder_name):
        save_folder = f'./{svae_folder_name}'
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        filename = f'{save_folder}/{self.img_number}.jpg'
        cv.imwrite(filename, self.result_img)
        self.img_number += 1

    def on(self):
        self.pipeline.start(self.config)
        while (True):
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            cv.imshow('liv', color_image)
            key = cv.waitKey(1) & 0xFF
            if key == ord('s'):
                cv.waitKey(0)
                break
            if self.stream_stop:
                break
            if self.iscaptured == True:
                self.capture()
                self.filtering()
                color_image_orign = np.asanyarray(self.color_frame.get_data())
                refPt = crop.crop(color_image_orign)
                self.input_image = color_image_orign[refPt[0][1]: refPt[1][1],
                                                    refPt[0][0]: refPt[1][0]]
                self.depth_roi = self.colorized_depth[refPt[0][1]: refPt[1][1],
                                                refPt[0][0]: refPt[1][0]]
                self.refPt = refPt
                refPt = None
                self.iscaptured = False
        cv.destroyAllWindows()

    def off(self):
        self.pipeline.stop()
        self.iscaptured = False

    def shotting(self): 
        self.iscaptured = True
        while self.input_image is None:
            continue
        self.capture_image = self.input_image
        width = self.input_image.shape[1]
        fx = 450 / width
        color_resize = cv.resize(self.input_image, dsize=(
            0, 0), fx=fx, fy=fx, interpolation=cv.INTER_LINEAR)
        depth_resize = cv.resize(self.depth_roi, dsize=(
            0, 0), fx=fx, fy=fx, interpolation=cv.INTER_LINEAR)

        self.depth_colormap = depth_resize.copy()
        self.color_image = color_resize.copy()
        self.get_depth_info()
        self.input_image = None
        return self.depth_colormap, self.color_image

    def get_depth_info(self):
        depth = np.asanyarray(self.aligned_depth_frame)
        depth_roi = depth[self.refPt[0][1]: self.refPt[1][1],
                          self.refPt[0][0]: self.refPt[1][0]]
        self.interpolation(depth_roi)

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

    # version 1
    # def run(self):
    #     self.depth_to_world()
    #     image_points = self.solve()
    #     self.remap(image_points)

    # versione 2
    # def run(self):
    #     self.depth_to_world()
    #     params = self.solve()
    #     parmas = self.optimaize(params)
    #     self.remap(parmas)

    # version 3
    def run(self):
        self.result_img = page_dewarp.run_dewarp(self.capture_image)
        width =self.result_img.shape[1]
        fx = 450 / width
        self.output_image = cv.resize(self.result_img, dsize=(0,0), fx=fx, fy=fx, interpolation=cv.INTER_AREA)


    def interpolation(self, depth):
        max_z = depth.max()
        depth = depth.astype(np.int16)
        depth *= -1
        depth += max_z
        self.depth = depth

    # version1
    # def remap(self, image_points):
    #     img_gray = cv.cvtColor(self.input_image, cv.COLOR_BGR2GRAY)
    #     x,y = img_gray.shape
    #     image_height_coords = image_points[:, 0, 0].reshape(
    #         (y, x)).astype(np.float32).T 
    #     image_width_coords = image_points[:, 0, 1].reshape(
    #         (y, x)).astype(np.float32).T 

    #     remapped = cv.remap(img_gray, image_height_coords,
    #                         image_width_coords, cv.INTER_CUBIC, None, cv.BORDER_REPLICATE)
    #     plt.imshow(remapped)
    #     plt.show()

    #version2 
    def remap(self, parmas):
        rvec = parmas[:3]
        tvec = parmas[3:6]
        image_points, _ = cv.projectPoints(
            self.objpoints, rvec, tvec, self.K, np.zeros(5))

        img_gray = cv.cvtColor(self.input_image, cv.COLOR_BGR2GRAY)
        x,y = img_gray.shape
        image_height_coords = image_points[:, 0, 0].reshape(
            (y, x)).astype(np.float32).T 
        image_width_coords = image_points[:, 0, 1].reshape(
            (y, x)).astype(np.float32).T 

        remapped = cv.remap(img_gray, image_height_coords,
                            image_width_coords, cv.INTER_CUBIC, None, cv.BORDER_REPLICATE)
        plt.imshow(remapped)
        plt.show()

    # version 2
    def solve(self):
        cubic_slopes = [0.0, 0.0]
        self.K = np.float32([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
        _, rvec, tvec = cv.solvePnP(
            self.world_list, self.image_list, self.K, np.zeros(5))

        parmas = np.hstack((np.array(rvec).flatten(),
                            np.array(tvec).flatten(),
                            np.array(cubic_slopes).flatten(),
                            ))

        return parmas

    def project(self, xy_coords, pvec):
        alpha, beta = tuple(pvec[6:8])
        poly = np.array([
            alpha + beta,
            -2*alpha - beta,
            alpha,
            0
        ])
        xy_coords = xy_coords.reshape((-1, 2))
        z_coords = np.polyval(poly, xy_coords[:, 0])
        self.objpoints = np.hstack((xy_coords, z_coords.reshape(-1, 1)))
        return self.objpoints

    def optimaize(self, parmas):
        def objective(pvec):
            objpoints = self.project(self.depth_hw_coords, pvec)
            return np.sum((self.world[:,2] - objpoints[:,2])**2)
        
        res = scipy.optimize.minimize(objective, parmas, method='Powell')
        return res.x

    # version 1
    # def solve(self):
    #     cubic_slopes = [0.0, 0.0]
    #     K = np.float32([[1, 0, 0],
    #                          [0, 1, 0],
    #                          [0, 0, 1]])
    #     _, rvec, tvec = cv.solvePnP(
    #         self.world_list, self.image_list, K, np.zeros(5))


    #     image_points, _ = cv.projectPoints(
    #         self.world, rvec, tvec, K, np.zeros(5))

    #     return image_points

    def depth_to_world(self):
        depth_height, depth_width = self.depth.shape
        depth_height_lin = np.linspace(0, depth_height - 1, depth_height)
        depth_width_lin = np.linspace(0, depth_width - 1, depth_width)
        self.depth_height_coords, self.depth_width_coords = np.meshgrid(
            depth_height_lin, depth_width_lin)
        self.depth_hw_coords = np.hstack((self.depth_height_coords.flatten().reshape((-1, 1)),
                                          self.depth_width_coords.flatten().reshape((-1, 1))))
        depth_coords = self.depth.T.flatten()
        self.world = np.hstack(
            (self.depth_hw_coords, depth_coords.reshape((-1, 1))))

        target_0 = self.world[np.where(self.world[:, 0] == 0)]
        target_bottom = self.world[np.where(self.world[:, 0] == depth_height - 1)]
        left_top = target_0[np.where(target_0[:, 1] == 0)][0]
        right_top = target_0[np.where(target_0[:, 1] == depth_width - 1)][0]
        left_bottom = target_bottom[np.where(target_bottom[:, 1] == 0)][0]
        right_bottom = target_bottom[np.where(target_bottom[:, 1] == depth_width - 1)][0]

        target_c = self.world[np.where(self.world[:, 0] == int((depth_height -1) / 2))]
        center = target_c[np.where(target_c[:, 1] == int((depth_width -1) / 2 ))][0]

        target_top_c = self.world[np.where(self.world[:, 0] == int((depth_height -1) / 4))]
        target_bottom_c = self.world[np.where(self.world[:, 0] == int((depth_height -1) * (3/4)))]
        center_top_left = target_top_c[np.where(target_top_c[:, 1] == int((depth_width -1) / 4 ))][0]
        center_top_right = target_top_c[np.where(target_top_c[:, 1] == int((depth_width -1) * (3/4) ))][0]
        center_bottom_left = target_bottom_c[np.where(target_bottom_c[:, 1] == int((depth_width -1) / 4 ))][0]
        center_bottom_right = target_bottom_c[np.where(target_bottom_c[:, 1] == int((depth_width -1) * (3/4) ))][0]
        self.world_list = np.array([
            left_top,  # page 왼쪽 상단
            left_bottom,  # page 왼쪽 아래
            right_top,  # page 오른쪽 상단
            right_bottom,  # page 오른쪽 하단
            center,  # center
            center_top_left,  # 왼쪽 상단 중간
            center_top_right,  # 오른쪽 상단 중간
            center_bottom_right,  # 오른쪽 하단 중간
            center_bottom_left,  # 왼쪽 하단 중간
        ], dtype=np.float32)

        self.image_list = np.array([
            left_top[:2],  # page 왼쪽 상단
            left_bottom[:2],  # page 왼쪽 아래
            right_top[:2],  # page 오른쪽 상단
            right_bottom[:2],  # page 오른쪽 하단
            center[:2],  # center
            center_top_left[:2],  # 왼쪽 상단 중간
            center_top_right[:2],  # 오른쪽 상단 중간
            center_bottom_right[:2],  # 오른쪽 하단 중간
            center_bottom_left[:2]  # 왼쪽 하단 중간
        ], dtype=np.float32)

    def visual(self):
        # book size
        depth_x, depth_y = self.depth.shape

        # graph axis x,y
        x = np.linspace(0, depth_x - 1, depth_x)
        y = np.linspace(0, depth_y - 1, depth_y)
        x = x[::10]
        y = y[::10]
        xx, yy = np.meshgrid(x, y)

        depth_sclae = self.depth[::10, ::10]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xx.T, yy.T, depth_sclae, s=1, cmap='Greens')
        plt.show()
