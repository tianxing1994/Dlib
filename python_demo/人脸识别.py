"""
人脸识别:
通过计算人脸的 128 维描述子特征向量来进行人脸识别.
通常如果两个特征向量的 Euclidean 距离小于 0.6 则它们是来自于同一个人脸, 否则是不同的人.
这里只展示人脸特征向量.

参考链接:
http://dlib.net/face_recognition.py.html

模型下载链接:
http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
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


def landmarks_to_ndarray(shape, dtype=np.int):
    """
    :param shape: 是 dlib 脸部特征检测的输出，一个 shape 里包含了脸部特征的 5 个点位.
    :param dtype:
    :return:
    """
    n = shape.num_parts
    coords = np.zeros(shape=(n, 2), dtype=dtype)
    for i in range(n):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


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


def show_landmarks_image(image, landmarks):
    landmarks = landmarks_to_ndarray(landmarks)

    x1, y1, x2, y2 = rect_to_bounding_box(rect)
    cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # w = x2 - x1
    # h = y2 - y1
    # print(f"bounding box w: {w}, h: {h}")

    for x, y in landmarks:
        cv.circle(image, (x, y), 2, (0, 0, 255), -1)
    show_image(image)
    return image


predictor_path = "../dataset/models/shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "../dataset/models/dlib_face_recognition_resnet_model_v1.dat"
faces_folder_path = "../dataset/johns"

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

image_path = "../dataset/face_detect_image/Tom_Cruise_avp_2014_4.jpg"
image = cv.imread(image_path)
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

rect2 = detector(image_rgb, 1)

for i, rect in enumerate(rect2):
    landmarks = sp(image_rgb, rect)

    show_landmarks_image(image, landmarks)

    # 在标定人脸关键点的情况下计算人脸的 128 维特征向量,
    # 通常如果两个特征向量的 Euclidean 距离小于 0.6 则它们是来自于同一个人脸, 否则是不同的人.
    # 这里只展示人脸特征向量.
    face_descriptor = facerec.compute_face_descriptor(image_rgb, landmarks)
    face_descriptor = np.array(face_descriptor)
    print(face_descriptor)

    # 这个函数也可以采用以下调用方式, 其中不代有参数 100, 意味着结果的 LFW 精度在 99.13%,
    # 而代有 100 的参数所对应的 LFW 精度为 99.38%, 但是执行速度要慢 100 倍,
    # face_descriptor = facerec.compute_face_descriptor(image_rgb, landmarks, 100, 0.25)
    face_descriptor = facerec.compute_face_descriptor(image_rgb, landmarks, 10, 0.25)
    face_descriptor = np.array(face_descriptor)
    print(face_descriptor)

    # 实际上, 第 3 个参数是指定将图像抖动/重采样(jitter/resample) 多少次,
    # 将其设置为 100 时, 它将在略有改动的人脸图像上执行 100 次描述子提取, 并返回其平均值,
    # 你也可以取一个中间值, 如 10, 这样执行速度就慢 10 倍, 对应的 LFW 精度为 99.3%.

    # 第 4 个参数(0.25) 是指在人脸图像周围的 padding. 如果 padding==0,
    # 则提取描述子的人脸图像将紧贴着 rect 所指定的大小截剪,
    # 设置较大的值则会有更宽松的截剪. 特别的, 当 padding=0.5 时, 会使截剪宽度增加一倍,
    # padding=1 截前宽度为 0 时的 3 倍, 以此类推.

    # compute_face_descriptor 还有一个重载函数可以接收对齐的图像作为输入值.
    # 请注意, 输入图像的大小必须为 150x150, 居中并缩放. (get_face_chip 的返回图像大小正是 150x150)

    # get_face_chip 根据 landmarks 对 image_rgb 图像做透视变换, 将人脸摆正,
    # 并从中截取人脸部份的图像, 然后将大小缩放至 150x150
    face_chip = dlib.get_face_chip(image_rgb, landmarks)
    show_image(face_chip)
    face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)
    face_descriptor = np.array(face_descriptor)
    print(face_descriptor)
