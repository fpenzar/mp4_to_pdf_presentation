# Importing all necessary libraries
from mimetypes import init
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
import cv2
import os
import shutil

# Read the video from specified path
root_dir = "/home/fpenzar/work/fer/code/utr/mp4_to_presentation"
all_mp4 = os.listdir(root_dir + "/mp4")
for mp4 in all_mp4:
    suffix = mp4.split('.')[0]
    cam = cv2.VideoCapture(root_dir + "/mp4/" + suffix + ".mp4")

    try:

        # creating a folder named data
        if not os.path.exists(root_dir + '/data'):
            os.makedirs(root_dir + '/data')

        # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame
    currentframe = 0
    frame_per_second = cam.get(cv2.CAP_PROP_FPS)
    score = 0
    prev_gray_image = None
    initial = True
    all_jpegs = []

    while (True):
        # reading from frame
        ret, frame = cam.read()

        # if video is still left continue creating images
        if ret:
            # take a screenshot every 0.7 seconds:
            if (currentframe % int(frame_per_second * 0.7) == 0):
                # calculate score {-1, 1} for current and previous valid picture (find how much they differ)
                if not initial:
                    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    score, diff = compare_ssim(prev_gray_image, new_gray, full=True)
                else:
                    initial = False
                if score < 0.92:
                    name = root_dir + '/data/frame' + str(currentframe) + '.jpg'
                    print('Creating...' + name)
                    # writing the extracted images
                    cv2.imwrite(name, frame)
                    # setting previous picture to current frame
                    prev_gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    all_jpegs.append(name)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

    # save jpgs to pdf
    image_list = []
    initial = True
    for jpg_name in all_jpegs:
        if not initial:
            image = Image.open(jpg_name)
            im = image.convert('RGB')
            image_list.append(im)
        else:
            initial = False

    image = Image.open(all_jpegs[0])
    im = image.convert('RGB')
    im.save(root_dir + "/pdf/" + suffix + ".pdf", save_all=True, append_images=image_list)

    # delete data directory
    shutil.rmtree(root_dir + "/data")