from os.path import join, splitext
import sys

LOG_DIR = join('log', splitext(sys.argv[0])[0])
TRAIN_DIR = join(LOG_DIR, 'train')
TEST_DIR = join(LOG_DIR, 'test')
N_TRAINS = 20
BATCH_SIZE = 100

WIDTH = 28
HEIGHT = 28
CHANNELS = 1
FLAT = WIDTH * HEIGHT * CHANNELS
N_CLASSES = 10
