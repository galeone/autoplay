#!/usr/bin/env python3
import sys
import time
from threading import Thread
import tensorflow as tf
from gi.repository import Playerctl, GLib
import numpy as np
import cv2
from ml import encoder, decoder, detect_face, model_fn, get_parser_fn


def facer(player):
    face_cascade_classifier = cv2.CascadeClassifier(
        "resources/haarcascade_frontalface_default.xml")
    thres = 0.015
    cap = cv2.VideoCapture(0)
    seconds = 5

    estimator = tf.estimator.Estimator(model_fn, "./model_dir/")
    input_ = tf.placeholder(tf.int8, shape=(None, None, None, 3))
    parser = get_parser_fn(tf.estimator.ModeKeys.PREDICT)
    parsed_input = parser(input_)
    reconstructions = decoder(encoder(parsed_input), parsed_input.shape[-1])
    loss = tf.losses.mean_squared_error(parsed_input, reconstructions)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, estimator.latest_checkpoint())

        paused = False
        squared_error_vals = []
        not_detected = 0
        while True:
            _, frame = cap.read()
            face_rect = detect_face(face_cascade_classifier, frame)
            if face_rect is not None:
                not_detected = 0
                x, y, w, h = face_rect
                face = frame[y:y + h, x:x + w]
                squared_error_val = sess.run(
                    loss, feed_dict={input_: np.expand_dims(face, axis=0)})
                squared_error_vals.append(squared_error_val)
                if len(squared_error_vals) >= seconds or paused:
                    mse = np.mean(np.array(squared_error_vals))
                    print(mse)
                    squared_error_vals.clear()
                    if mse < thres:
                        player.pause()
                    else:
                        player.play()
                        paused = False
            else:
                # there's no one in front of the camera for more than 10 seconds
                not_detected += 1
                if not_detected > 2 * seconds:
                    not_detected = 0
                    player.pause()
                    paused = True

            time.sleep(1)


def main():
    player = Playerctl.Player(player_name='vlc')
    vision = Thread(target=facer, args=(player,))
    main_loop = GLib.MainLoop()
    vision.start()
    main_loop.run()
    vision.join()


if __name__ == "__main__":
    sys.exit(main())
