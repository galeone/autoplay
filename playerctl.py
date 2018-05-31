#!/usr/bin/env python3
import tensorflow as tf
from gi.repository import Playerctl, GLib
from ml import encoder, get_input_fn


def model():
    input_ = tf.placeholder(tf.int32)
    input_fn = get_input_fn(input_)
    input_fn(1, tf.estimator.ModeKeys.PREDICT)


def main():
    player = Playerctl.Player(player_name='vlc')

    def on_play(player):
        # capture current user face and check if it has headphones on
        pass

    def on_pause(player):
        # wait for the user to get the headphones on again
        pass

    player.on('play', on_play)
    player.on('pause', on_pause)

    # wait for events
    main = GLib.MainLoop()
    main.run()


if __name__ == "__main__":
    sys.exit(main())
