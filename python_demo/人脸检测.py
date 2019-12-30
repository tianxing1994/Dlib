"""
人脸检测

参考链接:
http://dlib.net/cnn_face_detector.py.html
"""
import cv2 as cv
import dlib


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


model_path = "../dataset/models/mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(model_path)

image_path = "../dataset/face_detect_image/face_image1.jpg"
# 算法原来的图像应该是 RGB, 而 OpenCV 的图像格式是 BGR
image = cv.imread(image_path)

# 运行速度真的很慢.
# 第 2 个参数 1 表示将图像上采样一次, 再进行检测, 这样图像会变得更大.
faces = cnn_face_detector(image, 1)
for i, face in enumerate(faces):
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - face.rect.left()
    h = face.rect.bottom() - face.rect.top()
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

show_image(image)
