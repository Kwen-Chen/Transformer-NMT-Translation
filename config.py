# -- Paths --
DATA_DIR = 'data/raw'
CHECKPOINT_DIR = 'checkpoints'
LOGS_DIR = 'logs'

# -- Model Parameters --
D_MODEL = 512
NHEAD = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
MAX_LEN = 128

# -- Training Parameters --
NUM_EPOCHS = 35
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
CLIP_GRAD = 1.0
LOG_INTERVAL = 100

# -- Distributed Training --
# These can be overridden by command-line arguments
DISTRIBUTED = False
LOCAL_RANK = 0

# -- Other --
NUM_WORKERS = 4
