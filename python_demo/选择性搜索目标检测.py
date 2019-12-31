"""
选择性目标检测
检测图像中任何可能的物体区域,
其实现基于选择性搜索图像分割算法:
paper: Segmentation as Selective Search for Object Recognition by Koen E. A. van de Sande, et al.

通常, 您会将其用作对象检测管道的一部分. find_candidate_object_locations() 提名可能包含一个对象的框,
然后在每个框上运行一些昂贵的分类器并丢弃错误警报. 由于 find_candidate_object_locations() 仅生成数千个矩形,
因此它比扫描图像中所有可能的矩形要快得多.

参考链接:
http://dlib.net/find_candidate_object_locations.py.html
"""
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


image_path = '../dataset/face_detect_image/2009_004587.jpg'
image = cv.imread(image_path)
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

rects = []
dlib.find_candidate_object_locations(image_rgb, rects, min_size=500)

print("number of rectangles found {}".format(len(rects)))
for k, d in enumerate(rects):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))

    x1, y1, x2, y2 = rect_to_bounding_box(d)
    cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

show_image(image)
