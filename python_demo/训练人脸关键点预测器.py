"""
参考链接:
http://dlib.net/train_shape_predictor.py.html

此算法根据 paper 实现:
One Millisecond Face Alignment with an Ensemble of Regression Trees by Vahid Kazemi and Josephine Sullivan, CVPR 2014
"""
import os
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


def build_options():
    options = dlib.shape_predictor_training_options()
    # 这个算法有很多参数, 这里只设置其中的 3 个为不同的值, 其它采用默认值. 因为当前的训练集很小.
    # 将 oversampling 设为一个较大的值 300, 以提升训练集大小.
    options.oversampling_amount = 300

    # 通过显示地增加正则化 (使 nu 为较小值) 和使用深度较小的树来减小模型的容量.
    options.nu = 0.05
    options.tree_depth = 2
    options.be_verbose = True
    return options


def train_model(training_xml_path, options):
    # dlib.train_shape_predictor() 执行训练任务. 最后会将训练好的结果存储到 predictor.dat 文件中.
    # 模型训练的输入是一个 XML 文件, 其中列出了图片及其脸部关键点的标记.
    dlib.train_shape_predictor(training_xml_path, "../dataset/temp/predictor.dat", options)
    return


def demo1():
    # 训练模型
    faces_folder = "../dataset/faces"
    training_xml_path = os.path.join(faces_folder, "training_with_face_landmarks.xml")
    options = build_options()
    train_model(training_xml_path, options)
    return


def eval_model(training_xml_path, testing_xml_path):
    # dlib.test_shape_predictor() 函数计算预测关键点与真实值之间的距离平均值作为模型评估.
    print("\nTraining accuracy: {}".format(
        dlib.test_shape_predictor(training_xml_path, "../dataset/temp/predictor.dat")))

    # 评估在测试集的表现.
    print("Testing accuracy: {}".format(
        dlib.test_shape_predictor(testing_xml_path, "../dataset/temp/predictor.dat")))
    return


def demo2():
    faces_folder = "../dataset/faces"
    training_xml_path = os.path.join(faces_folder, "training_with_face_landmarks.xml")
    testing_xml_path = os.path.join(faces_folder, "testing_with_face_landmarks.xml")
    eval_model(training_xml_path, testing_xml_path)
    return


def demo3():
    predictor = dlib.shape_predictor("../dataset/temp/predictor.dat")
    detector = dlib.get_frontal_face_detector()

    image_path = "../dataset/face_detect_image/face_image1.jpg"
    image = cv.imread(image_path)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    rects = detector(image_rgb, 1)
    for rect in rects:
        x1, y1, x2, y2 = rect_to_bounding_box(rect)
        cv.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 225), thickness=2)

        landmarks = predictor(image_rgb, rect)
        landmarks = landmarks_to_ndarray(landmarks)
        for x, y in landmarks:
            cv.circle(image, (x, y), 2, (0, 0, 255), -1)

    show_image(image)
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    demo3()
