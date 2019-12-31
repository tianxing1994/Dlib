"""
参考链接:
http://dlib.net/face_detector.py.html
"""
import cv2 as cv
import dlib


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    image_path = "../dataset/face_detect_image/face_image1.jpg"
    image = cv.imread(image_path)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    rects = detector(image_rgb, 1)
    for i, rect in enumerate(rects):
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()
        cv.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)

    show_image(image)
