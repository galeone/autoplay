import sys
import os
import tensorflow as tf
import numpy as np
import cv2


def encoder(inputs):
    with tf.variable_scope("encoder"):
        l2_regularizer = tf.contrib.layers.l2_regularizer(1e-5)

        fc1 = tf.layers.dense(
            inputs,
            512,
            activation=tf.nn.tanh,
            kernel_initializer=tf.initializers.variance_scaling(),
            kernel_regularizer=l2_regularizer,
            name="fc1")

        fc2 = tf.layers.dense(
            fc1,
            256,
            activation=tf.nn.tanh,
            kernel_initializer=tf.initializers.variance_scaling(),
            kernel_regularizer=l2_regularizer,
            name="fc2")
        latent = tf.layers.dense(
            fc2,
            128,
            activation=tf.nn.tanh,
            kernel_initializer=tf.initializers.variance_scaling(),
            kernel_regularizer=l2_regularizer,
            name="latent")

    return latent


def decoder(latent, units):
    with tf.variable_scope("decoder"):
        l2_regularizer = tf.contrib.layers.l2_regularizer(1e-5)

        fc1 = tf.layers.dense(
            latent,
            256,
            activation=tf.nn.tanh,
            kernel_initializer=tf.initializers.variance_scaling(),
            kernel_regularizer=l2_regularizer,
            name="fc1")
        fc2 = tf.layers.dense(
            fc1,
            512,
            activation=tf.nn.tanh,
            kernel_initializer=tf.initializers.variance_scaling(),
            kernel_regularizer=l2_regularizer,
            name="fc2")

        reconstruction = tf.layers.dense(
            fc2,
            units,
            activation=tf.nn.tanh,
            kernel_initializer=tf.initializers.variance_scaling(),
            name="reconstruction")

    return reconstruction


def autoencoder(inputs):
    with tf.variable_scope("autoencoder"):
        reconstructions = decoder(encoder(inputs), inputs.shape[-1])

    return reconstructions


def main():
    dataset = "dataset"
    exists = os.path.exists(dataset)

    empty = not os.listdir(dataset) if exists else True
    if not exists or empty:
        if not exists:
            os.makedirs(dataset)

        face_cascade_classifier = cv2.CascadeClassifier(
            "resources/haarcascade_frontalface_default.xml")

        cap = cv2.VideoCapture(0)
        dataset_size = 0
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade_classifier.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE)

            if faces.size:
                original = np.copy(frame)

                # Draw a rectangle around the faces
                bigger_id = 0
                bigger_area = 0
                for (x, y, w, h), idx in enumerate(faces):
                    area = w * h
                    if area > bigger_area:
                        bigger_id = idx
                        bigger_area = w * h

                x, y, w, h = faces[bigger_id]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the resulting frame
                cv2.imshow('Video', frame)

                if cv2.waitKey() & 0xFF == ord('s'):
                    face = original[y:y + h, x:x + w]
                    cv2.imwrite(
                        os.path.join(dataset,
                                     str(dataset_size) + ".png"), face)
                    dataset_size += 1
            else:
                # Display the resulting frame
                cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def model_fn(features, labels, mode):
        reconstructions = autoencoder(features)

        loss = tf.losses.mean_squared_error(features, reconstructions)
        loss = loss + tf.losses.get_regularization_loss()

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Create Optimiser
            optimizer = tf.train.AdamOptimizer()

            # Create training operation
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(
                mode, predictions=reconstructions, loss=loss, train_op=train_op)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                eval_metric_ops={
                    "rmse":
                    tf.metrics.root_mean_squared_error(features,
                                                       reconstructions)
                })

        assert mode == tf.estimator.ModeKeys.PREDICT
        return tf.estimator.EstimatorSpec(mode, predictions=reconstructions)

    def _input_fn(batch_size, mode, num_epochs=1):

        def parser(filename):
            shape = (80, 80)
            image = tf.image.decode_jpeg(tf.read_file(filename))
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize_images(image, shape)

            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = (image - .5) * 2.  # [-1, 1]

            if mode == tf.estimator.ModeKeys.TRAIN:
                image = tf.image.random_flip_left_right(image)
                image = image + tf.random_normal(tf.shape(image), stddev=1e-3)

            image = tf.clip_by_value(image, -1., 1.)
            image = tf.reshape(image, (shape[0] * shape[1],))
            return image

        def input_fn():
            filename_dataset = tf.data.Dataset.list_files(
                "{}/*.jpg".format(dataset))
            image_dataset = filename_dataset.map(parser)
            if mode == tf.estimator.ModeKeys.TRAIN:
                image_dataset = image_dataset.repeat(num_epochs).shuffle(10000)

            image_dataset = image_dataset.batch(batch_size)
            iterator = image_dataset.make_one_shot_iterator()
            return iterator.get_next()

        return input_fn

    tf.logging.set_verbosity(tf.logging.INFO)
    config = tf.estimator.RunConfig("./model_dir/", save_summary_steps=100)
    model = tf.estimator.Estimator(model_fn, config=config)
    model.train(_input_fn(32, tf.estimator.ModeKeys.TRAIN, 1000))
    model.evaluate(_input_fn(32, tf.estimator.ModeKeys.EVAL))


if __name__ == "__main__":
    sys.exit(main())
