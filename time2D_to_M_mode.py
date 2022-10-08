import os
from glob import glob

import cv2
import numpy as np


def gen_col_B_mode(img_list: list, target_col_idx: int, inverse=False):
    first_list_img = cv2.imread(img_list[0], cv2.IMREAD_GRAYSCALE)
    rows, cols = first_list_img.shape
    output = np.zeros((len(img_list), cols))

    for i, img_path in enumerate(img_list):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if inverse:
            output[-1-i] = img[target_col_idx]
        else:
            output[i] = img[target_col_idx]
    output = np.expand_dims(output, axis=-1).astype(np.uint8)

    return output

def gen_row_B_mode(img_list: list, target_row_idx: int, inverse=False):
    first_list_img = cv2.imread(img_list[0], cv2.IMREAD_GRAYSCALE)
    rows, cols = first_list_img.shape
    output = np.zeros((len(img_list), rows))

    for i, img_path in enumerate(img_list):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.transpose((1, 0))
        if inverse:
            output[-1-i] = img[target_row_idx]
        else:
            output[i] = img[target_row_idx]
    output = np.expand_dims(output, axis=-1).astype(np.uint8)

    return output


def main_proc():

    first_img_name = '000.png'
    inverse_vertical = False

    specified_x = 370; specified_y = 305

    target_folder = './pending'
    save_folder = './result'

    img_ext = os.path.splitext(first_img_name)[1]
    img_digit_begin = int(os.path.splitext(first_img_name)[0])

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    first_imgs = glob(target_folder + '/**/' + first_img_name, recursive=True)

    for first_img in first_imgs:
        current_path = os.path.dirname(first_img)
        img_list = glob(current_path + '/*' + img_ext, recursive=False)
        img_list.sort()

        save_path = save_folder + '/' + current_path.replace(target_folder, '').replace(' ', '_')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        img_B_mode_cols = gen_col_B_mode(img_list, specified_y, inverse=inverse_vertical)
        img_B_mode_rows = gen_row_B_mode(img_list, specified_x, inverse=inverse_vertical)

        cv2.imwrite(save_path+'/B_mode_cols.png', img_B_mode_cols)
        cv2.imwrite(save_path+'/B_mode_rows.png', img_B_mode_rows)

def UT():

    first_img_name = '000.png'
    inverse_vertical = False

    specified_x = 370; specified_y = 305

    target_test_folder = './pending/sample/database_2d_simulations_uffc/GE Vingmed Ultrasound/A2C/normal'

    img_ext = os.path.splitext(first_img_name)[1]
    img_digit_begin = int(os.path.splitext(first_img_name)[0])

    img_list = glob(target_test_folder + '/*' + img_ext, recursive=False)
    img_list.sort()

    img_B_mode_cols = gen_col_B_mode(img_list, specified_y, inverse=inverse_vertical)
    img_B_mode_rows = gen_row_B_mode(img_list, specified_x, inverse=inverse_vertical)

    cv2.imshow('img_B_mode_cols', img_B_mode_cols)
    cv2.waitKey()
    cv2.destroyAllWindows()

    cv2.imshow('img_B_mode_rows', img_B_mode_rows)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main_proc()
    # UT()
