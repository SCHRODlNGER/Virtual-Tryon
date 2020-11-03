import cv2
import numpy as np
import glob
import os

def grabcut(input_dir = "./data/cloth", output_dir = "./data/cloth-mask"):
    for image_path in glob.glob(os.path.join(input_dir, "*.*g")):
        image_name = image_path.split("/")[-1].split(".")[0]
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        th, im_th = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV);

        im_floodfill = im_th.copy()
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0,0), 255);
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = im_th | im_floodfill_inv

        cv2.imwrite(os.path.join(output_dir , image_name  + ".jpg"), im_out)


if __name__ == "__main__":
    grabcut()    