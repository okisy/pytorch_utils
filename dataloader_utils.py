import numpy as np
import cv2
from scipy.ndimage.interpolation import rotate

def load_image(img_path):
    im_bgr = cv2.imread(img_path)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    h,w,c = im_rgb.shape
    return im_rgb, (h,w)

def resize_to_determined_size(img, max_size=1024):
    '''
    img:
        numpy array of [h, w, c]
    '''
    h, w, c = img.shape
    ratio = max_size / max(h,w)
    new_h = h * ratio
    new_w = w * ratio
    img = cv2.resize(img, (int(new_w), int(new_h)))

    return img

### from https://github.com/clovaai/CRAFT-pytorch/blob/master/imgproc.py ###
def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    '''
    in_img:
        numpy array of [h,w,c]
    # should be RGB order    
    # RGB values vary from 0-255
    '''
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    '''
    in_img:
        numpy array of [h,w,c]
    # should be RGB order    
    # RGB values are already normalized
    '''
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


### basic data augmentation ###
### from https://github.com/xkumiyu/numpy-data-augmentation ###
def horizontal_flip(image1, rate=0.5):
    if np.random.rand() < rate:
        image1 = image1[:, ::-1, :]
    return image1

def vertical_flip(image1, rate=0.5):
    if np.random.rand() < rate:
        image1 = image1[::-1, :, :]
    return image1

def random_crop(image1, image2, crop_size=(224, 224)):    
    h, w, _ = image1.shape

    # 0~(400-224)の間で画像のtop, leftを決める
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])

    # top, leftから画像のサイズである224を足して、bottomとrightを決める
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    # 決めたtop, bottom, left, rightを使って画像を抜き出す
    image1 = image1[top:bottom, left:right, :]
    return image1

def scale_augmentation(image1, scale_range=(256, 400), crop_size=224):
    scale_size = np.random.randint(*scale_range)
    image1 = cv2.resize(image1, (scale_size, scale_size))
    image1 = random_crop(image1, (crop_size, crop_size))
    return image1

def random_rotation(image1, angle_range=(-5, 5)):
    h, w, _ = image1.shape
    angle = np.random.randint(*angle_range)
    image1 = rotate(image1, angle)
    image1 = cv2.resize(image1, (h, w))
    return image1


def horizontal_flip_2imgs(image1, image2, rate=0.5):
    if np.random.rand() < rate:
        image1 = image1[:, ::-1, :]
        image2 = image2[:, ::-1, :]
    return image1, image2

def vertical_flip_2imgs(image1, image2, rate=0.5):
    if np.random.rand() < rate:
        image1 = image1[::-1, :, :]
        image2 = image2[::-1, :, :]
    return image1, image2

def random_crop_2imgs(image1, image2, crop_size=(224, 224)):
    assert image1.shape[:2] == image2.shape[:2]    
    h, w, _ = image1.shape

    # 0~(400-224)の間で画像のtop, leftを決める
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])

    # top, leftから画像のサイズである224を足して、bottomとrightを決める
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    # 決めたtop, bottom, left, rightを使って画像を抜き出す
    image1 = image1[top:bottom, left:right, :]
    image2 = image2[top:bottom, left:right, :]
    return image1, image2

def scale_augmentation_2imgs(image1, image2, scale_range=(256, 400), crop_size=224):
    scale_size = np.random.randint(*scale_range)
    image1 = cv2.resize(image1, (scale_size, scale_size))
    image2 = cv2.resize(image2, (scale_size, scale_size))
    image1, image2 = random_crop_2imgs(image1, image2, (crop_size, crop_size))
    return image1, image2

def random_rotation_2imgs(image1, image2, angle_range=(-5, 5)):
    assert image1.shape[:2] == image2.shape[:2]
    h, w, _ = image1.shape
    angle = np.random.randint(*angle_range)
    image1 = rotate(image1, angle)
    image2 = rotate(image2, angle)
    image1 = cv2.resize(image1, (h, w))
    image2 = cv2.resize(image2, (h, w))
    return image1, image2
