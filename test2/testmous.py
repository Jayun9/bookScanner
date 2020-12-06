from cv2 import cv2 as cv
import numpy as np


# 검은색 바탕을 생성합니다. 마우스 콜백함수를 바인드 합니다.
img = cv.imread('1.jpg')
cv.imshow('img',img)
cv.waitKey()
cv.destroyWindow()
