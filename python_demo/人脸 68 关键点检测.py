"""
参考链接:
http://dlib.net/face_landmark_detection.py.html

模型下载链接:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""
import cv2 as cv
import dlib
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def rect_to_bounding_box(rect):
    """
    :param rect: 是 dlib 脸部区域检测的输出
    :return:
    """
    x1 = rect.left()
    y1 = rect.top()
    x2 = rect.right()
    y2 = rect.bottom()
    return x1, y1, x2, y2


def landmarks_to_ndarray(shape, dtype=np.int):
    """
    :param shape: 是 dlib 脸部特征检测的输出，一个 shape 里包含了脸部特征的 n 个点位.
    :param dtype:
    :return:
    """
    n = shape.num_parts
    coords = np.zeros(shape=(n, 2), dtype=dtype)
    for i in range(n):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


predictor_path = "../dataset/models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

image_path = "../dataset/face_detect_image/football_man.jpg"
# image_path = "../dataset/face_detect_image/Tom_Cruise_avp_2014_4.jpg"
img = cv.imread(image_path)
image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 第 2 个参数 1 表示将图像上采样一次, 再进行检测, 这样图像会变得更大, 能检测到的人脸也可能会变多.
rects = detector(image_rgb, 2)
for i, rect in enumerate(rects):
    landmarks = predictor(image_rgb, rect)
    landmarks = landmarks_to_ndarray(landmarks)

    x1, y1, x2, y2 = rect_to_bounding_box(rect)
    cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for x, y in landmarks:
        cv.circle(img, (x, y), 1, (0, 0, 255), -1)

show_image(img)
