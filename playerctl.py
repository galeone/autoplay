#!/usr/bin/env python3
import sys
import time
from threading import Thread, Lock
import tensorflow as tf
from gi.repository import Playerctl, GLib
import numpy as np
import cv2
from ml import encoder, decoder, detect_face, model_fn, get_parser_fn


class WebcamVideoStream:

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self):
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()


def facer(player):
    face_cascade_classifier = cv2.CascadeClassifier(
        "resources/haarcascade_frontalface_default.xml")
    thres = 0.5
    cap = cv2.VideoCapture(0)
    #cap = WebcamVideoStream()

    tick = 20

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
            #frame = cap.read()
            face_rect = detect_face(face_cascade_classifier, frame)
            if face_rect is not None:
                not_detected = 0
                x, y, w, h = face_rect
                x -= 30
                y -= 30
                w += 30
                h += 30
                face = frame[y:y + h, x:x + w]
                squared_error_val = sess.run(
                    loss, feed_dict={input_: np.expand_dims(face, axis=0)})
                squared_error_vals.append(squared_error_val)
                if len(squared_error_vals) >= tick or paused:
                    mse = np.mean(np.array(squared_error_vals))
                    print(mse)
                    squared_error_vals.clear()
                    if mse > thres:
                        player.pause()
                    else:
                        player.play()
                        paused = False
            else:
                # there's no one in front of the camera for more than 2*tick
                not_detected += 1
                if not_detected > 2 * tick:
                    not_detected = 0
                    player.pause()
                    paused = True
            time.sleep(0.15)


def main():
    player = Playerctl.Player(player_name='spotify')
    vision = Thread(target=facer, args=(player,))
    main_loop = GLib.MainLoop()
    vision.start()
    main_loop.run()
    vision.join()


if __name__ == "__main__":
    sys.exit(main())
