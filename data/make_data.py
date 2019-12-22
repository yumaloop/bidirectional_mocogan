import os 
import csv
import pathlib
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from position_functions import posFunc

def concat_image(video_seq, frame_size = 20):
    for i in range(frame_size):
        img = video_seq[i]
        height = img.shape[0]
        if i == 0:
            img_concat = img
        else:
            img_concat = np.concatenate([img_concat, np.full((height, 2),255)], 1)
            img_concat = np.concatenate([img_concat, img], 1)
    return img_concat

def make_frame(digit_img, motion_label=0, frame_size=20, width=64, height=64):
    output_video = np.full((frame_size, width, height), 0)

    for frame_id in range(frame_size):    
        output_img = output_video[frame_id]

        # numpy to pil
        digit_img_pil = Image.fromarray(np.uint8(digit_img))
        output_img_pil = Image.fromarray(np.uint8(output_img))

        # rotate  
        angle = np.random.rand(1) * -15 + 15 # angle ~ U(-15, 15)
        digit_img_pil = digit_img_pil.rotate(angle)

        position = posFunc(frame_id, motion_label)
        # position = posFunc_Right2Left_Down(frame_id)

        # paste digit_img to output_img
        output_img_pil.paste(digit_img_pil, position)

        # pil to numpy
        output_img = np.asarray(output_img_pil)
        output_video[frame_id] = output_img

    return output_video


def main(frame_size=20):
    # path settings
    label_file_path = './labels.csv'
    video_dir_path = './video'
    if os.path.exists(label_file_path):
        os.remove(label_file_path)
        pathlib.Path(label_file_path).touch()
    if not os.path.exists(video_dir_path):
        os.mkdir(video_dir_path)

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # make labe.csv
    p_empty = pathlib.Path('./labels.csv')
    if not p_empty.exists(): p_empty.touch()
    
    for i, digit_img in enumerate(tqdm(x_test)):
        for motion_label in range(8):
            output_video = make_frame(digit_img, motion_label)
            
            # video (npz)
            index = i * 8 + motion_label
            npy_path   = "./video/" + str(index) 
            np.save(npy_path, output_video)

            # label (csv)
            with open('./labels.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([index, motion_label])

if __name__ == '__main__':
    main()
