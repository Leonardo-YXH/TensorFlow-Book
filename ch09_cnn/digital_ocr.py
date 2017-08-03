import cv2
import mnist_predict
import numpy as np
import os

GRAY_THRESHOLD = 80
X_BORDER=6
Y_BORDER=6

def grayify(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def thresholding_inv(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if gray>48:then 255;else 0
    # if backgroud is dark,then cv2.THRESH_BINARY,else cv2.THRESH_BINARY_INV
    ret, bin = cv2.threshold(gray, GRAY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    return cv2.medianBlur(bin, 3)


class RectImage(object):
    def __init__(self, pos, image):
        """
        :param pos: up-left cordinate(x,y)
        :param image: source image
        """
        self.pos = pos
        self.image = image

    def get_position(self):
        return self.pos

    def get_image(self):
        return self.image


def format_mnist(image):
    """
    convet 20*20 to 28*28 by adding border
    :param image: 
    :return: 
    """
    top = np.zeros((4, 28))
    left = np.zeros((20, 4))
    image = np.hstack((left, image))
    image = np.hstack((image, left))
    image = np.vstack((top, image))
    image = np.vstack((image, top))
    return image


def cmp_rect_image(o1, o2):
    if o1.get_position()[0] > o2.get_position()[0]:
        return 1
    elif o1.get_position()[0] < o2.get_position()[0]:
        return -1
    else:
        return 0

def filled_image(image,x_border,y_border):
    """
    
    :param image: 2-D array
    :param x_border: 
    :param y_border: 
    :return: 
    """
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j]==0:#black
                left=False
                right=False
                top=False
                down=False
                for t in range(x_border):
                    if j-t>=0:
                        if image[i][j-t]==255:#white
                            left=True
                    else:
                        break
                for t in range(x_border):
                    if j + t < image.shape[1]:
                        if image[i][j + t] == 255:
                            right = True
                    else:
                        break
                for t in range(x_border):
                    if i - t >=0:
                        if image[i - t][j] == 255:  # white
                            top = True
                    else:
                        break
                for t in range(x_border):
                    if i + t < image.shape[0]:
                        if image[i + t][j] == 255:
                            down = True
                    else:
                        break

                if (left and right) or (top and down):
                    image[i][j]=2550
def retrive_digits(fileName, flatten=True):
    img = cv2.imread(fileName)
    #gray = grayify(img)
    thres = thresholding_inv(img)
    thres_filled=thres.copy()
    #filled_image(thres_filled,X_BORDER,Y_BORDER)
    cv2.imwrite(fileName + 'thres.png', thres)
    cv2.imshow(" ",thres_filled)
    cv2.waitKey(0)
    _, contours, _ = cv2.findContours(thres_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours)==0:
        return
    images = []
    rects = []
    sum_w = 0
    sum_h = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        sum_w += w
        sum_h += h
        rects.append((x, y, w, h))

    mean_w = sum_h / len(rects)
    min_w = mean_w / 3.0
    mean_h = sum_h / len(rects)
    min_h = mean_h / 3.0
    for (x, y, w, h) in rects:
        if w < min_w :
            if h < min_h:
                continue
            else:
                cropped = thres[y:y + h, x-int(min_h):x + int(min_h)] #label 1
        else:
            cropped = thres[y:y + h, x:x + w]
        resized = cv2.resize(cropped, (20, 20))  # mnist data format
        resized = format_mnist(resized)
        rect_img = RectImage((x, y), resized)
        images.append(rect_img)

    images = sorted(images, key=lambda x: x.get_position()[0], reverse=False)
    mnist_predict1 = mnist_predict.mnist_data_predict()
    #mnist_predict1=mnist_predict.mnist_data_predict(meta_graph="./lcd_data/log/my_model.ckpt.meta",checkpoint_path="./lcd_data/log")
    number=1
    for image in images:
        img_data = image.get_image()
        number%=10
        fn=fileName.replace(".","_"+str(number)+".")
        #cv2.imwrite(fn, img_data)
        number+=1
        img_data = np.reshape(img_data, (784))
        img_data = img_data.astype(np.float32)
        img_data = np.multiply(img_data, 1.0 / 255.0)
        cv2.imshow("a", image.get_image())
        mnist_predict1.predict(img_data)
        cv2.waitKey(0)

def preprocess_img(root_dir):
    cfg=[[0,0],[0,1],[0,0],[0,0],[1,1],[1,1],[1,0]]
    for dir in os.listdir(root_dir):
        if os.path.isfile(dir):
            retrive_digits(dir)

def resize():
    for i in range(10):
        fn="lcd_data/lcd_03_"+str(i)+".png"
        img=cv2.resize(cv2.imread(fn),(28,28))
        cv2.imwrite(fn,img)
if __name__ == "__main__":
    retrive_digits("images/digit_sample_03.png")
    #resize()
