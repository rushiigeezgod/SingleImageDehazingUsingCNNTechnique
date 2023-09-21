import glob
import random

import matplotlib.pyplot as plt
import tensorflow as tf

from customcode.dehaze_net import gman_net

k_init = tf.keras.initializers.random_normal(stddev=0.008, seed=101)
regularizer = tf.keras.regularizers.L2(1e-4)
b_init = tf.constant_initializer()

net = gman_net(k_init, b_init,regularizer)


def evaluate(net, test_img_path):
    return_display_list = []
    test_img = glob.glob(test_img_path + "/*.jpg")
    random.shuffle(test_img)

    for img in test_img:
        img = tf.io.read_file(img)
        img = tf.io.decode_jpeg(img, channels=3)

        if img.shape[1] > img.shape[0]:
            img = tf.image.resize(img, size=(412, 548), antialias=True)
        if img.shape[1] < img.shape[0]:
            img = tf.image.resize(img, size=(412, 548), antialias=True)

        img = img / 255.0
        img = tf.expand_dims(img, axis=0)  # transform input image from 3D to 4D ###

        dehaze = net(img, training=False)

        plt.figure(figsize=(80, 80))

        display_list = [img[0], dehaze[0]]  # make the first dimension zero
        title = ["Hazy Image", "Dehazed Image"]

        # for i in range(2):
        #     plt.subplot(1, 2, i+1)
        #     plt.title(title[i], fontsize = 65, y = 1.045)
        #     plt.imshow(display_list[i])
        #     plt.axis('off')
        #
        # plt.show()
        return_display_list.append(display_list)

    return return_display_list


def dehazed_app(test_net,folder_path):
    
    return evaluate(test_net, folder_path)

