import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from lib.config.settings import EMBED_DIM
from .layers import Conv1d, Conv2d, get_norm

class SignModel_(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        # EfficientNet-B0 feature 추출기
        # self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        # self.backbone._fc = nn.Identity()  # 분류 헤드 제거
        
        # avg Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # self.feature_dim = 1280  # EfficientNet-B0 출력 차원
        self.feature_dim = 1680
        self.embed_dim = EMBED_DIM  # config에서 가져옴 (512)
        self.num_layers = 3
        self.num_heads = 8
        self.head_dim = self.embed_dim // self.num_heads  # 64
        
        # 특징 차원 변환
        self.feature_projection = nn.Linear(self.feature_dim, self.embed_dim)
        
        # Multi-Head Self-Attention
        self.mhsa_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=0.2,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])
        
        # Feed-Forward Networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 4),
                nn.GELU(),
                nn.Linear(self.embed_dim * 4, self.embed_dim),
                nn.Dropout(0.2)
            ) for _ in range(self.num_layers)
        ])
        
        # Layer Normalization
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.embed_dim) for _ in range(self.num_layers * 2)  # attention + ffn용
        ])
        
        # 최종 분류기
        self.classifier = nn.Linear(self.embed_dim, vocab_size)
        
        self._init_weights()

    def forward(self, x):
        # keypoints 데이터 처리 (B, T, V, D) 형태 가정
        # batch, padded_frame_cnt, keypoint_shape, channel(x, y, confidence)
        if x.dim() == 5:  # (B, T, V, K, D) keypoints 데이터
            batch_size, seq_len, num_views, kepoint_data, coord_dim = x.shape
            x = x.view(batch_size, seq_len, -1)  # (B, T, V, K*D)
            # 특징 차원 변환
            x = self.feature_projection(x)  # (B, T, embed_dim)
            
        # elif x.dim() == 5:  # (B, T, C, H, W) 이미지 데이터  
        #     batch_size, seq_len, C, H, W = x.shape
        #     x = x.view(batch_size * seq_len, C, H, W)
        #     print(f"after view : {x.shape}")
        #     # EfficientNet-B0 특징 추출
        #     features = self.backbone.extract_features(x)
        #     features = self.gap(features).squeeze(-1).squeeze(-1)
        #     features = features.view(batch_size, seq_len, self.feature_dim)
        #     x = self.feature_projection(features)
            
        # 트랜스포머 레이어 적용
        for i in range(self.num_layers):
            # Multi-Head Self-Attention
            attn_out, _ = self.mhsa_layers[i](x, x, x)
            x = self.norm_layers[i*2](x + attn_out)  # residual connection + norm
            
            # Feed-Forward Network
            ffn_out = self.ffn_layers[i](x)
            x = self.norm_layers[i*2+1](x + ffn_out)  # residual connection + norm
        
        # 전역 평균 풀링 후 분류
        x = x.mean(dim=1)  # (B, embed_dim)
        output = self.classifier(x)  # (B, vocab_size)
        
        return output

    def _init_weights(self):
        # 트랜스포머 부분 초기화
        for module in [self.feature_projection, self.classifier]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # FFN 레이어 초기화
        for ffn in self.ffn_layers:
            for layer in ffn:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

class SignModel(nn.Module):

    def __init__(self, vocab):
        super().__init__()
        activation = nn.ReLU()
        norm_2d = "BN_2d"
        norm_1d = "BN_1d"

        # 2d feature extraction
        self.conv1 = Conv2d(
            3, 32, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm2d(32)
        )  # (3, 224, 224) -> (32, 224, 224)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # (32, 112, 112)

        self.conv2 = Conv2d(
            32, 64, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm2d(64)
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)  # (64, 56, 56)

        self.conv3 = Conv2d(
            64, 64, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm2d(64)
        )
        self.conv4 = Conv2d(
            64, 128, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm2d(128)
        )
        self.pool4 = nn.MaxPool2d(2, stride=2)  # (128, 28, 28)

        self.conv5 = Conv2d(
            128, 128, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm2d(128)
        )
        self.conv6 = Conv2d(
            128, 256, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm2d(256)
        )
        self.pool6 = nn.MaxPool2d(2, stride=2)  # (256, 14, 14)

        self.conv7 = Conv2d(
            256, 256, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm2d(256)
        )
        self.conv8 = Conv2d(
            256, 512, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm2d(512)
        )
        self.pool8 = nn.MaxPool2d(2, stride=2)  # (512, 7, 7)

        self.conv9 = Conv2d(
            512, 512, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm2d(512)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # (512, 1, 1)

        # temporal encode
        self.tconv1 = Conv1d(
            512, 512, 5, stride=1, padding=2, activation=activation, norm=nn.BatchNorm1d(512)
        )
        self.tpool1 = nn.MaxPool1d(2, stride=2)

        self.tconv2 = Conv1d(
            512, 512, 5, stride=1, padding=2, activation=activation, norm=nn.BatchNorm1d(512)
        )
        self.tpool2 = nn.MaxPool1d(2, stride=2)

        self.tconv3 = Conv1d(
            512, 1024, 3, stride=1, padding=1, activation=activation, norm=nn.BatchNorm1d(1024)
        )

        # classification
        self.classifier = nn.Linear(1024, len(vocab))

        # init
        self.init_layers()

    def init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, T, C, H, W) -> (B*T, C, H, W)
        # features: (B*T, 512, 1, 1) -> (B, T, 512)
        # after t1: (B, T//4, 512)
        # after t2: (B, T//4, 1024)
        # how to handle linear layer ? 1d input shape?
        _, clip_length, _, _, _ = x.shape
        """
        temporal_group = int(clip_length / self.num_groups)
        results = []
        for t in range(0, clip_length, temporal_group):
            # t_slice = x[:, t:t + temporal_group]  # (B, group, C, H, W)
            results.append(self.extract_feature(x[:, t:t + temporal_group]))  # (B, C, group)
        x = torch.cat(results, dim=2)
        assert x.size(2) == clip_length
        """
        x = self.extract_feature(x)
        # temporal encoding
        x = self.tpool1(self.tconv1(x))
        x = self.tpool2(self.tconv2(x))
        x = self.tconv3(x)  # (batch, 1024, T//4)

        # classifier
        x = x.transpose(1, 2)  # (batch, T//4, 1024)
        x = self.classifier(x)  # (batch, T//4, C)
        return x

    def extract_feature(self, x):
        batch_size, clip_length, C, H, W = x.shape
        x = x.view(batch_size * clip_length, C, H, W)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool4(self.conv4(self.conv3(x)))
        x = self.pool6(self.conv6(self.conv5(x)))
        x = self.pool8(self.conv8(self.conv7(x)))
        x = self.avg_pool(self.conv9(x))  # (B*T, 512, 1, 1)
        x = x.view(x.shape[:2]).view(batch_size, clip_length, -1).transpose(1, 2)  # (B, C, T)
        return x
