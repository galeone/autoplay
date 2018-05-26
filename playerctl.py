#!/usr/bin/env python3
from gi.repository import Playerctl, GLib


def main():
    player = Playerctl.Player(player_name='vlc')

    def on_metadata(player, e):
        if 'xesam:artist' in e.keys() and 'xesam:title' in e.keys():
            print('Now playing:')
            print('{artist} - {title}'.format(
                artist=e['xesam:artist'][0], title=e['xesam:title']))

    def on_play(player):
        # capture current user face with headphones on
        print('Playing at volume {}'.format(player.props.volume))

    def on_pause(player):
        #
        print('Paused the song: {}'.format(player.get_title()))

    player.on('play', on_play)
    player.on('pause', on_pause)
    player.on('metadata', on_metadata)

    # wait for events
    main = GLib.MainLoop()
    main.run()


if __name__ == "__main__":
    sys.exit(main())
