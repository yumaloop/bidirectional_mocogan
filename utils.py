import torch
import numpy as np

def concat_image(video):
    # video: 3D-np.array (frame_size, height, width)
    for i, img in enumerate(video):
        if i == 0:
            img_concat = img
        else:
            # img_concat = np.concatenate([img_concat, np.full((img.shape[0], 2), 0.5), img], 1)
            img_concat = np.concatenate([img_concat, np.full((img.shape[0], 2), 1.0), img], 1)
    return img_concat


def make_save_image(videos_fake, num_sample=5):
    """
    video_fake: torch.Tensor (num, frame_len, height, width)
    """
    videos_fake_arr = videos_fake[0:num_sample].cpu().detach().numpy()
    # videos_real_arr = videos_real[0:num_sample].cpu().detach().numpy()
    videos_fake_arr = videos_fake_arr.reshape(num_sample, 20, 64, 64)
    # videos_real_arr = videos_real_arr.reshape(num_sample, 20, 64, 64)

    for i in range(num_sample):
        img_fake = concat_image(videos_fake_arr[i])
        # img_real = concat_image(videos_real_arr[i])
        vertical_pad = np.full((2, img_fake.shape[1]), 1.0)
        img_both = np.vstack((img_fake, vertical_pad))
        if i == 0:
            img_gen = img_both
        else:
            img_gen = np.vstack((img_gen, img_both))
    return img_gen
