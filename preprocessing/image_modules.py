import numpy as np
import tensorflow as tf


def to_writeable_array(tf_img):
    img = tf.keras.preprocessing.image.img_to_array(tf_img)
    img = np.copy(img)
    
    img_list = []
    
    for i in range(len(img)):
        row = []
        for j in range(len(img)):
            pixel = np.array([
            np.uint8(img[i][j][0] *  255),
            np.uint8(img[i][j][1] *  255),
            np.uint8(img[i][j][2] *  255)
            ])
            
            row.append(pixel)

            img[i][j][1] = np.uint8(img[i][j][1] *  255)
            img[i][j][2] = np.uint8(img[i][j][2] *  255)
            img[i][j][0] = np.uint8(img[i][j][0] *  255)
            img[i][j] = img[i][j].astype(np.uint8)
    
        img_list.append(row)
    img_np = np.array(img_list)
    return img_np
    
def write_image_to_path(dest_path):
    write_able = to_writeable_array(img_save)
    img = im.fromarray(write_able)
    img.save(dest_path)