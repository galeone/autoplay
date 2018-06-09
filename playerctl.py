#!/usr/bin/env python3
import sys
import tensorflow as tf
from gi.repository import Playerctl, GLib
import numpy as np
import cv2
from ml import encoder, decoder, detect_face, model_fn, get_parser_fn


def main():
    face_cascade_classifier = cv2.CascadeClassifier(
        "resources/haarcascade_frontalface_default.xml")

    thres = 0.015

    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)

    estimator = tf.estimator.Estimator(model_fn, "./model_dir/")
    input_ = tf.placeholder(tf.int8, shape=(None, None, None, 3))
    parser = get_parser_fn(tf.estimator.ModeKeys.PREDICT)
    parsed_input = parser(input_)
    reconstructions = decoder(encoder(parsed_input), parsed_input.shape[-1])
    mse = tf.losses.mean_squared_error(parsed_input, reconstructions)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, estimator.latest_checkpoint())

        faces = []
        play_not_detected = 0

        def on_play(player):
            # capture current user face and check if it has headphones on
            _, frame = cap.read()
            face_rect = detect_face(face_cascade_classifier, frame)
            print(face_rect)
            if face_rect is not None:
                x, y, w, h = face_rect
                face = frame[y:y + h, x:x + w]
                faces.append(face)
                if len(faces) >= fps:
                    mse_val = sess.run(mse, feed_dict={input_: np.array(faces)})
                    faces.clear()
                    if mse_val < thres:
                        player.pause()
            else:
                play_not_detected += 1
                if play_not_detected > fps:
                    play_not_detected = 0
                    player.pause()

        def on_pause(player):
            # wait for the user to get the headphones on again
            pass

        player = Playerctl.Player(player_name='vlc')
        player.on('play', on_play)
        player.on('pause', on_pause)

        # wait for events
        main_loop = GLib.MainLoop()
        main_loop.run()


if __name__ == "__main__":
    sys.exit(main())
