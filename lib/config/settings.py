import os
import torch

# 프로젝트 경로
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TRAINED_MODEL_DIR = os.path.join(PROJECT_ROOT, 'trained_model')
MODEL_SAVE_PATH = os.path.join(TRAINED_MODEL_DIR, 'best_model.pt')

# 학습 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 50

MAX_SEQ_LEN = 600
KEYPOINT_DIM = 21*3*2 + 33*3  # 손(21관절×3좌표×2손)+얼굴(33관절×3좌표)
EMBED_DIM = 512

VOCAB_SPECIAL_TOKENS = {'<pad>': 0, '<unk>': 1}

# # 훈련 설정
# BATCH_SIZE = 64
# EPOCHS = 100
# DROPOUT = 0.3
# OPTIMIZER = 'Adam'
# LOSS_FUNCTION = 'CrossEntropyLoss'

# 트랜스포머 설정
NUM_TRANSFORMER_LAYERS = 3
NUM_ATTENTION_HEADS = 3
ATTENTION_HEAD_DIM = 64
EMBED_DIM = 512