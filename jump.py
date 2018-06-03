import cv2
# import numpys as np
import matplotlib.pyplot as plt

VERSION = "1.1.4"
scale = 0.25

template = cv2.imread('./resource/image/character.png')
template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
template_size = template.shape[:2]

def search(img):
    result = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    cv2.rectangle(
        img,
        (min_loc[0], min_loc[1]),
        (min_loc[0] + template_size[1], min_loc[1] + template_size[0]),
        (255, 0, 0),
        4)
    return img, min_loc[0] + template_size[1] / 2, min_loc[1] +  template_size[0]



def update_data():
    global src_x, src_y

    # img = cv2.imread('./autojump.png')
    img = cv2.imread("./resource/image/1.jpg")
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    img, src_x, src_y = search(img)
    return img, src_x, src_y

img, src_x, src_y=update_data()

print(src_x)
print("==========")
print(src_y)
cv2.namedWindow("Image")
cv2.imshow("Image", img)
cv2.waitKey(0)
# im = plt.imshow(img, animated=True)