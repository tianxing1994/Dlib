import cv2 as cv
import dlib


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


filename = "mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(filename)

f = "dataset/face1.jpg"
image = cv.imread(f)
show_image(image)
dets = cnn_face_detector(image, 1)

print(dets)


