import sys
import os
import tensorflow as tf
import numpy as np
import cv2


def encoder(inputs):
    with tf.variable_scope("encoder"):
        l2_regularizer = tf.contrib.layers.l2_regularizer(1e-5)

        latent = tf.layers.dense(
            inputs,
            128,
            activation=tf.nn.tanh,
            kernel_initializer=tf.initializers.variance_scaling(),
            kernel_regularizer=l2_regularizer,
            name="latent")

    return latent


def decoder(latent, units):
    with tf.variable_scope("decoder"):
        reconstruction = tf.layers.dense(
            latent,
            units,
            activation=tf.nn.tanh,
            kernel_initializer=tf.initializers.variance_scaling(),
            name="reconstruction")

    return reconstruction


def autoencoder(inputs):
    with tf.variable_scope("autoencoder"):
        reconstructions = decoder(encoder(inputs), inputs.shape[-1])

    return reconstructions


def get_input_fn(data_source):

    use_placeholder = isinstance(
        data_source, tf.Tensor) and 'placeholder' in data_source.name.lower()

    def input_fn(batch_size, mode, num_epochs=1):

        def parser(image):
            shape = (80, 80)
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

        def _input_fn():
            if not use_placeholder:
                filename_dataset = tf.data.Dataset.list_files(
                    "{}/*.png".format(data_source))
                image_dataset = filename_dataset.map(lambda filename: tf.image.decode_png(tf.read_file(filename)))
                image_dataset = image_dataset.map(parser)
                if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                    image_dataset = image_dataset.repeat(num_epochs)
                    if mode == tf.estimator.ModeKeys.TRAIN:
                        image_dataset = image_dataset.shuffle(1000)

                image_dataset = image_dataset.batch(batch_size)
                iterator = image_dataset.make_one_shot_iterator()
                return iterator.get_next()

            #data_source is placeholder
            return parser(data_source)

        return _input_fn

    return input_fn


def model_fn(features, labels, mode):

    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    latent = encoder(features)

    average_latent = ema.average(latent)
    reconstructions = decoder(features, features.shape[-1])

    loss = tf.losses.mean_squared_error(features, reconstructions)
    loss = loss + tf.losses.get_regularization_loss()

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Create Optimiser
        optimizer = tf.train.AdamOptimizer(1e-5)

        # Create training operation
        opt_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        with tf.control_dependencies([opt_op]):
            train_op = ema.apply([latent])

        return tf.estimator.EstimatorSpec(
            mode, predictions=reconstructions, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops={
                "rmse":
                tf.metrics.root_mean_squared_error(features, reconstructions)
            })

    assert mode == tf.estimator.ModeKeys.PREDICT
    return tf.estimator.EstimatorSpec(mode, predictions=reconstructions)


def detect_face(face_cascade_classifier, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = np.array(
        face_cascade_classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE))

    if faces.size:
        bigger_id = 0
        bigger_area = 0
        for idx, (_, _, w, h) in enumerate(faces):
            area = w * h
            if area > bigger_area:
                bigger_id = idx
                bigger_area = w * h

        return faces[bigger_id]  # (x,y,w,h)

    return None


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
            _, frame = cap.read()
            face_rect = detect_face(face_cascade_classifier, frame)

            if face_rect is not None:
                original = np.copy(frame)
                x, y, w, h = face_rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the resulting frame
                cv2.imshow('Video', frame)

                key = cv2.waitKey() & 0xFF

                if key == ord('s'):
                    face = original[y:y + h, x:x + w]
                    cv2.imwrite(
                        os.path.join(dataset,
                                     str(dataset_size) + ".png"), face)
                    dataset_size += 1
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    break
            else:
                # Display the resulting frame
                cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    tf.logging.set_verbosity(tf.logging.INFO)
    config = tf.estimator.RunConfig(
        "./model_dir/", save_summary_steps=10, save_checkpoints_steps=100)
    model = tf.estimator.Estimator(model_fn, config=config)

    input_fn = get_input_fn(dataset)
    model.train(input_fn(32, tf.estimator.ModeKeys.TRAIN, 500))
    model.evaluate(input_fn(32, tf.estimator.ModeKeys.EVAL))


if __name__ == "__main__":
    sys.exit(main())
