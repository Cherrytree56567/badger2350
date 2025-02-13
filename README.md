# Badger 2350 <!-- omit in toc -->

[![MicroPython Firmware](https://github.com/CherryTree56567/badger2350/actions/workflows/micropython.yml/badge.svg?branch=main)](https://github.com/CherryTree56567/badger2350/actions/workflows/micropython.yml)

## Firmware, Examples & Documentation <!-- omit in toc -->

Badger 2350 are maker-friendly all-in-one badge wearables, featuring a 2.9", 296x128 pixel, monochrome e-paper display.

- [Install](#install)
  - [Badger 2350](#badger-2350)

## Install

Grab the latest release from [https://github.com/CherryTree56567/badger2350/releases/latest](https://github.com/CherryTree56567/badger2350/releases/latest)

There are four .uf2 files to pick from.

:warning: Those marked `with-badger-os` contain a full filesystem image that will overwrite both the firmware *and* filesystem of your Badger:

* pimoroni-badger2350-vX.X.X-micropython-with-badger-os.uf2 

The regular builds just include the firmware, and leave your files alone:

* pimoroni-badger2350-vX.X.X-micropython.uf2 

###  Badger 2350

1. Connect your Badger 2350 to your computer using a USB A to C cable.

2. Reset your device into bootloader mode by holding BOOT/USR and pressing the RST button next to it.

3. Drag and drop one of the `badger2350` .uf2 files to the "RP2350" drive that appears.

4. Your device should reset and, if you used a `with-badger-os` variant, show the Badger OS Launcher.

### Credits
Dungeon Game from https://github.com/judah4/badger-dungeon
Chess Game from https://github.com/niutech/chess-badger2040
