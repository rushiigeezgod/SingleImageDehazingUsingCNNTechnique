import time

import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error

from customcode.output_img import display_img


def train_model(epochs, train, val, net, train_loss_tracker, val_loss_tracker, optimizer):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,), end=" ")
        start_time_epoch = time.time()
        start_time_step = time.time()

        # training loop

        for step, (train_batch_orig, train_batch_haze) in enumerate(train):
            with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
                train_logits = net(train_batch_haze, training=True)  # Give training data as input.
                loss = mean_squared_error(
                    train_batch_orig, train_logits
                )  # Calculate the loss between original image and output of training

            grads = tape.gradient(loss, net.trainable_weights)  # Compute multiple gradients over the same computation.
            optimizer.apply_gradients(zip(grads, net.trainable_weights))

            train_loss_tracker.update_state(train_batch_orig, train_logits)  # Update the loss after every epoch
            if step == 0:
                print("[", end="")
            if step % 64 == 0:
                print("=", end="")

        print("]", end="")
        print("  -  ", end="")
        print("Training Loss: %.4f" % (train_loss_tracker.result()), end="")

        # validation loop

        for step, (val_batch_orig, val_batch_haze) in enumerate(val):
            val_logits = net(val_batch_haze, training=False)
            val_loss_tracker.update_state(val_batch_orig, val_logits)

            if step % 32 == 0:
                display_img(net, val_batch_haze, val_batch_orig)

        print("  -  ", end="")
        print("Validation Loss: %.4f" % (val_loss_tracker.result()), end="")
        print("  -  ", end=" ")
        print("Time taken: %.2fs" % (time.time() - start_time_epoch))

        net.save("trained_model")  # save the model(variables, weights, etc).
        train_loss_tracker.reset_states()  # Reset the loss for new loss from next epoch.
        val_loss_tracker.reset_states()
