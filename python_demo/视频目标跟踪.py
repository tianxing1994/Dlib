"""
视频目标跟踪

参考链接:
http://dlib.net/correlation_tracker.py.html
"""
import os
import glob
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


video_folder = "../dataset/video_frames"
tracker = dlib.correlation_tracker()

for i, frame_path in enumerate(sorted(glob.glob(os.path.join(video_folder, "*.jpg")))):
    img = dlib.load_rgb_image(frame_path)
    frame = cv.imread(frame_path)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    if i == 0:
        # 设置需要跟踪的目标.
        tracker.start_track(frame_rgb, dlib.rectangle(74, 67, 112, 153))
    else:
        # Else we just attempt to track from the previous frame
        tracker.update(frame_rgb)

    rect = tracker.get_position()
    x1, y1, x2, y2 = rect_to_bounding_box(rect)
    cv.rectangle(frame, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=(0, 0, 255), thickness=1)
    show_image(frame)
