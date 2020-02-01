"""
参考链接:
http://dlib.net/train_object_detector.py.html

目标:
原目标是, 检测出图像中螺丝孔的位置, 并判断有哪些螺丝已安装, 哪些没有安装.
这里的实现可以检测出未安装的螺丝的位置.

算法原理是 HOG 特征 + SVM 分类. 但是我用 opencv 的 HOG + SVM 没有成功.
"""
import os
import cv2 as cv
import dlib


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


def build_options():
    # train_simple_object_detector() 函数的参数.
    options = dlib.simple_object_detector_training_options()

    # 左右翻转变化, 数据增强.
    options.add_left_right_image_flips = True

    # 训练器是支持向量机, 所以有一个 C 参数, 取较大的值将使其在训练集上拟合的更好, 但有可能过拟合.
    # 你应该多尝试一些 C 值以找到最好的, 而不是将其设成 5 而已.
    options.C = 5

    # 告诉训练器你有多少个 CPU, 以提高训练速度.
    options.num_threads = 4
    # 如果为 True train_simple_object_detector() 将在模型训练时打印出详细信息.
    options.be_verbose = True
    return options


def train_model(training_xml_path, options):
    # 该函数执行训练任务, 并将最后的检测器存放到 detector.svm 文件中.
    # 输入是一个 XML 文件, 列出了训练集中的图片及人脸位置的标注.
    # 要创建你自己的训练集, 你可以使用 tools/imglab 文件夹下的 imglab 工具.
    # 这一次是直接使用 dlib 提供的 xml 文件.
    dlib.train_simple_object_detector(training_xml_path, "../dataset/temp/detector.svm", options)
    return


def demo1():
    # 不使用 XML 文件来训练模型.
    # 最后, 我们不一定需要用 XML 文件来做为训练器 train_simple_object_detector() 函数的输入.
    # 也可以如下这样:
    # 将图片对象放入一个列表中
    faces_folder = "../dataset/other/luosi"
    images = [dlib.load_rgb_image(faces_folder + '/luosi.jpg')]

    # box 的标记.
    boxes_img1 = ([dlib.rectangle(left=80, top=256, right=112, bottom=288),
                   dlib.rectangle(left=211, top=323, right=243, bottom=355),
                   dlib.rectangle(left=601, top=250, right=633, bottom=282),
                   dlib.rectangle(left=738, top=317, right=770, bottom=349)])

    # 将标记数据放入一个列表.
    boxes = [boxes_img1]

    detector2 = dlib.train_simple_object_detector(images, boxes, build_options())

    # 将训练好的模型保存.
    detector2.save('../dataset/other/luosi/temp/detector.svm')

    # 通过以下方法来查看训练模型的效果.
    print("\nTraining accuracy: {}".format(
        dlib.test_simple_object_detector(images, boxes, detector2)))
    return


def demo2():
    # 在任意图像中执行人脸检测任务.
    # 加载检测器
    detector = dlib.simple_object_detector("../dataset/other/luosi/temp/detector.svm")

    image_path = "../dataset/other/luosi/luosi.jpg"
    image = cv.imread(image_path)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    rects = detector(image_rgb)
    for rect in rects:
        x1, y1, x2, y2 = rect_to_bounding_box(rect)
        cv.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)

    show_image(image)
    return


def demo3():
    # 我们可以查看我们训练好的人脸 HOG 滤波器
    detector = dlib.fhog_object_detector("../dataset/other/luosi/temp/detector.svm")
    win_det = dlib.image_window()
    win_det.set_image(detector)
    dlib.hit_enter_to_continue()
    return


if __name__ == '__main__':
    # demo1()
    demo2()
    # demo3()
