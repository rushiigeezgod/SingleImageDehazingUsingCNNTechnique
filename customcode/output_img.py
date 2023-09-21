import matplotlib.pyplot as plt

from customcode.dehaze_net import gman_net

k_init = tf.keras.initializers.random_normal(stddev=0.008, seed=101)
regularizer = tf.keras.regularizers.L2(1e-4)
b_init = tf.constant_initializer()

model = gman_net(k_init, b_init, regularizer)

# function to display output.


def display_img(model, hazy_img, orig_img):
    dehazed_img = model(hazy_img, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [hazy_img[0], orig_img[0], dehazed_img[0]]
    title = ["Hazy Image", "Ground Truth", "Dehazed Image"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis("off")

    plt.show()
