"""
人脸五点检测: 鼻尖和四个眼角.

参考链接:
http://dlib.net/face_alignment.py.html
https://my.oschina.net/wujux/blog/1622777
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
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def landmarks_to_ndarray(shape, dtype=np.int):
    """
    :param shape: 是 dlib 脸部特征检测的输出，一个 shape 里包含了脸部特征的 5 个点位.
    :param dtype:
    :return:
    """
    coords = np.zeros(shape=(5, 2), dtype=dtype)
    for i in range(5):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


predictor_path = '../dataset/models/shape_predictor_5_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
landmarks_detector = dlib.shape_predictor(predictor_path)


image_path = "../dataset/face_detect_image/face_image1.jpg"
image = cv.imread(image_path)
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# 检测脸部
face_rects = detector(image_rgb, 1)

if len(face_rects) == 0:
    print("No face found.")
    exit(0)

# 查找脸部位置
faces = dlib.full_object_detections()
for face_rect in face_rects:
    face_landmarks = landmarks_detector(image_rgb, face_rect)
    landmarks_ndarray = landmarks_to_ndarray(face_landmarks)

    x, y, w, h = rect_to_bounding_box(face_rect)
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for x, y in landmarks_ndarray:
        cv.circle(image, (x, y), 2, (0, 0, 255), -1)

show_image(image)
