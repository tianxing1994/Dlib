"""
人脸抖动增强
其实就是透视变换, 数据增强.

参考链接:
http://dlib.net/face_jitter.py.html

模型下载链接:
http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
"""
import cv2 as cv
import dlib


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


# 人脸 5 点检测模型.
predictor_path = "../dataset/models/shape_predictor_5_face_landmarks.dat"
image_path = "../dataset/face_detect_image/Tom_Cruise_avp_2014_4.jpg"

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

image = cv.imread(image_path)
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
rects = detector(image_rgb)

face_rects_list = list()
for rect in rects:
    # sp(img, rect) 返回值: dlib.full_object_detection 对象,
    # 通过 .part 方法获取关键点坐标: ret.part(0).x, ret.part(0).y,
    # 通过 .num_parts 属性获取关键点的总数.
    face_rects_list.append(sp(image, rect))

# 获取检测到的人脸位置的 bounding box 截图.
image_face = dlib.get_face_chip(image, face_rects_list[0], size=320)
show_image(image_face)

# 图像增强, 但不改变颜色
jittered_face_images = dlib.jitter_image(image_face, num_jitters=5)
for jittered_image in jittered_face_images:
    show_image(jittered_image)

# 图像增强, 并增加光照变化.
jittered_face_images = dlib.jitter_image(image_face, num_jitters=5, disturb_colors=True)
for jittered_image in jittered_face_images:
    show_image(jittered_image)
