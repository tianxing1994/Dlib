"""
参考链接:
http://dlib.net/train_object_detector.py.html
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

    # 给人脸增加左右翻转变化
    options.add_left_right_image_flips = True

    # 训练器是支持向量机, 所以有一个 C 参数, 取较大的值将使其在训练集上拟合的更好, 但有可能过拟合.
    # 你应该多尝试一些 C 值以找到最好的, 而不是将其设成 5 而已.
    options.C = 5

    # 告诉训练器你有多少个 CPU, 以提高训练速度.
    options.num_threads = 4
    options.be_verbose = True
    return options


def train_model(training_xml_path, options):
    # 该函数执行训练任务, 并将最后的检测器存放到 detector.svm 文件中.
    # 输入是一个 XML 文件, 列出了训练集中的图片及人脸位置的标注.
    # 要创建你自己的训练集, 你可以使用 tools/imglab 文件夹下的 imglab 工具.
    # 这一次是直接使用 dlib 提供的 xml 文件.
    dlib.train_simple_object_detector(training_xml_path, "../dataset/temp/detector.svm", options)
    return


def eval_model():
    # 查看训练模型在训练集和测试集的效果.
    faces_folder = "../dataset/faces"
    training_xml_path = os.path.join(faces_folder, "training.xml")
    testing_xml_path = os.path.join(faces_folder, "testing.xml")

    # 现在我们有了一个人脸检测器, 我们来测试它的准确率, 召回率, 平均精度.
    print("Training accuracy: {}".format(
        dlib.test_simple_object_detector(training_xml_path, "../dataset/temp/detector.svm")))
    print("Testing accuracy: {}".format(
        dlib.test_simple_object_detector(testing_xml_path, "../dataset/temp/detector.svm")))
    return


def demo1():
    # 训练人脸检测器
    faces_folder = "../dataset/faces"
    training_xml_path = os.path.join(faces_folder, "training.xml")
    options = build_options()
    train_model(training_xml_path, options)
    return


def demo2():
    # 在任意图像中执行人脸检测任务.
    # 加载检测器
    detector = dlib.simple_object_detector("../dataset/temp/detector.svm")

    image_path = "../dataset/face_detect_image/face_image1.jpg"
    image = cv.imread(image_path)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    rects = detector(image_rgb)
    for rect in rects:
        x1, y1, x2, y2 = rect_to_bounding_box(rect)
        cv.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)

    show_image(image)
    return


def demo3():
    # 假如我们训练了多个检测器, 我们可以将它们作为一个组来执行任务.
    detector1 = dlib.fhog_object_detector("../dataset/temp/detector.svm")
    detector2 = dlib.fhog_object_detector("../dataset/temp/detector.svm")

    # 将需要执行的检测器放到一个列表中.
    detectors = [detector1, detector2]

    image_path = "../dataset/face_detect_image/face_image1.jpg"
    image = cv.imread(image_path)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    [boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(detectors=detectors,
                                                                                 image=image_rgb,
                                                                                 upsample_num_times=1,
                                                                                 adjust_threshold=0.0)
    for i in range(len(boxes)):
        print("detector {} found box {} with confidence {}.".format(detector_idxs[i], boxes[i], confidences[i]))
    return


def demo4():
    # 不使用 XML 文件来训练模型.
    # 最后, 我们不一定需要用 XML 文件来做为训练器 train_simple_object_detector() 函数的输入.
    # 也可以如下这样:
    # 将图片对象放入一个列表中
    faces_folder = "../dataset/faces"
    images = [dlib.load_rgb_image(faces_folder + '/2008_002506.jpg'),
              dlib.load_rgb_image(faces_folder + '/2009_004587.jpg')]

    # 人脸 box 的标记.
    boxes_img1 = ([dlib.rectangle(left=329, top=78, right=437, bottom=186),
                   dlib.rectangle(left=224, top=95, right=314, bottom=185),
                   dlib.rectangle(left=125, top=65, right=214, bottom=155)])
    boxes_img2 = ([dlib.rectangle(left=154, top=46, right=228, bottom=121),
                   dlib.rectangle(left=266, top=280, right=328, bottom=342)])

    # 将标记数据放入一个列表.
    boxes = [boxes_img1, boxes_img2]

    detector2 = dlib.train_simple_object_detector(images, boxes, build_options())

    # 将训练好的模型保存.
    detector2.save('../dataset/temp/detector.svm')

    # 通过以下方法来查看训练模型的效果.
    print("\nTraining accuracy: {}".format(
        dlib.test_simple_object_detector(images, boxes, detector2)))
    return


def demo5():
    # 我们可以查看我们训练好的人脸 HOG 滤波器
    detector = dlib.fhog_object_detector("../dataset/temp/detector.svm")
    win_det = dlib.image_window()
    win_det.set_image(detector)
    dlib.hit_enter_to_continue()
    return


if __name__ == '__main__':
    # demo1()
    demo2()
    # demo3()
    # demo4()
    # demo5()
