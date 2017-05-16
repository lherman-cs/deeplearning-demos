from os.path import join, splitext
import sys

MODEL_NAME = splitext(sys.argv[0])[0]
LOG_DIR = join('log', MODEL_NAME)
TRAIN_DIR = join(LOG_DIR, 'train')
TEST_DIR = join(LOG_DIR, 'test')
CHECKPOINTS_DIR = 'checkpoints'
N_TRAINS = 10000
BATCH_SIZE = 100

WIDTH = 28
HEIGHT = 28
CHANNELS = 1
FLAT = WIDTH * HEIGHT * CHANNELS
N_CLASSES = 10
