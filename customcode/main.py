import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from customcode.dehaze_net import gman_net
from customcode.path_dataloader import data_path, dataloader
from customcode.train import train_model

# Hyperparameters
epochs = 5
batch_size = 4
k_init = tf.keras.initializers.random_normal(stddev=0.008, seed=101)
regularizer = tf.keras.regularizers.L2(1e-4)
b_init = tf.constant_initializer()

train_data, val_data = data_path(
    orig_img_path='/content/drive/MyDrive/GT1',
    hazy_img_path='/content/drive/MyDrive/hazy1',
)

train, val = dataloader(train_data, val_data, batch_size)

optimizer = Adam(learning_rate=1e-4)  # we are using Adam optimizer.
net = gman_net(k_init, b_init,regularizer)

train_loss_tracker = tf.keras.metrics.MeanSquaredError(name="train loss")  # We are using MSE as loss metrics.
val_loss_tracker = tf.keras.metrics.MeanSquaredError(name="val loss")

# Call the training function.
train_model(epochs, train, val, net, train_loss_tracker, val_loss_tracker, optimizer)
