import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torchsummary import summary


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )  # Embedding dim으로 변환하며 패치크기의 커널로 패치크기만큼 이동하여 이미지를 패치로 분할 할 수 있음.

    def forward(self, x):
        x = self.proj(x)  # (batch_size, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # 세번째 차원부터 끝까지 flatten (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5  # 1 / root(self.head_dim)
        '''
        # 나중에 query와 key를 곱하고 softmax를 취하기전에 scale factor로 나눠주는데 이 scale factor의 역할은 
        query @ key 의 값이 커지게 되면 softmax 함수에서의 기울기 변화가 거의 없는 부분으로 가기때문에 gradient vanishing
        문제를 해결하려면 scaling을 해주어야 한다고 Attention is all you need 논문에서 주장
         
        '''
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches+1, dim*3)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_samples, n_patches+1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, n_samples, n_heads, n_patches+1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 각각의 n_heads끼리 query, key, value로 나눔
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches+1)  dot product를 위한 transpose
        # dot product를 통해 query와 key사이의 유사도를 구함
        dp = (q @ k_t) * self.scale  # (n_samples, n_heads, n_patches+1, n_patches+1)  @: dot product (2x1)@(1x2)=(2x2)
        attn = dp.softmax(dim=-1)  # attention (n_samples, n_heads, n_patches+1, n_patches+1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches+1, head_dim)
        # 원래 차원으로 되돌림.
        weighted_avg = weighted_avg.transpose(1, 2)  # (n_samples, n_patches+1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # concat (n_samples, n_patches+1, dim)

        x = self.proj(weighted_avg)  # linear projection (n_samples, n_patches+1, dim)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Encoder(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super(Encoder, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.hidden_features = int(dim * mlp_ratio)  # 3072(MLP size)
        self.mlp = MLP(
            in_features=dim,
            hidden_features= self.hidden_features,
            out_features=dim,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class Aux_MLP(nn.Module):
    def __init__(self, in_features, hidden_features ,out_features,n_classes, p=0.):
        super(Aux_MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)
        
        self.norm = nn.LayerNorm(out_features, eps=1e-6)
        self.head = nn.Linear(out_features, n_classes)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.drop(x)
        
        x = self.norm(x)
        cls_token_final = x[:, 0]  # just tje CLS token
        x = self.head(cls_token_final)
        return x
    
class SDiT_Base(nn.Module):
    def __init__(
            self,
            img_size=384,
            patch_size=16,
            in_chans=3,
            n_classes=1000,
            embed_dim=768,
            depth=12,
            n_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.):
        super(SDiT_Base, self).__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.n_classes = n_classes
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=p)
        
        self.block1 = Encoder(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
        self.block2 = Encoder(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
        self.block3 = Encoder(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
        self.block4 = Encoder(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
        self.block5 = Encoder(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
        self.block6 = Encoder(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
        self.block7 = Encoder(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
        self.block8 = Encoder(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
        self.block9 = Encoder(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
        self.block10 = Encoder(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
        self.block11 = Encoder(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
        self.block12 = Encoder(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
        
        self.aux_mlp1 = Aux_MLP(
            in_features=embed_dim,
            hidden_features= 64,
            out_features=32,
            n_classes=self.n_classes,
        )
        self.aux_mlp2 = Aux_MLP(
            in_features=embed_dim,
            hidden_features= 64,
            out_features=32,n_classes=self.n_classes,
        )
        self.aux_mlp3 = Aux_MLP(
            in_features=embed_dim,
            hidden_features= 64,
            out_features=32,n_classes=self.n_classes,
        )
        self.aux_mlp4 = Aux_MLP(
            in_features=embed_dim,
            hidden_features= 64,
            out_features=32,n_classes=self.n_classes,
        )
        self.aux_mlp5 = Aux_MLP(
            in_features=embed_dim,
            hidden_features= 64,
            out_features=32,n_classes=self.n_classes,
        )
        self.aux_mlp6 = Aux_MLP(
            in_features=embed_dim,
            hidden_features= 64,
            out_features=32,n_classes=self.n_classes,
        )
        self.aux_mlp7 = Aux_MLP(
            in_features=embed_dim,
            hidden_features= 64,
            out_features=32,n_classes=self.n_classes,
        )
        self.aux_mlp8 = Aux_MLP(
            in_features=embed_dim,
            hidden_features= 64,
            out_features=32,n_classes=self.n_classes,
        )
        self.aux_mlp9 = Aux_MLP(
            in_features=embed_dim,
            hidden_features= 64,
            out_features=32,n_classes=self.n_classes,
        )
        self.aux_mlp10 = Aux_MLP(
            in_features=embed_dim,
            hidden_features= 64,
            out_features=32,n_classes=self.n_classes,
        )
        self.aux_mlp11 = Aux_MLP(
            in_features=embed_dim,
            hidden_features= 64,
            out_features=32,n_classes=self.n_classes,
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)  # (n_samples, n_patches, embed_dim)
        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1+n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, 1+n_patches, embed_dim)
        x = self.pos_drop(x)

        # for block in self.blocks:
        #     x = block(x)  # (n_samples, 577, 768)
        
        x = self.block1(x)
        middle_output1 = self.aux_mlp1(x)
        
        x = self.block2(x)
        middle_output2 = self.aux_mlp2(x)
        
        x = self.block3(x)
        middle_output3 = self.aux_mlp3(x)
        
        x = self.block4(x)
        middle_output4 = self.aux_mlp4(x)
        
        x = self.block5(x)
        middle_output5 = self.aux_mlp5(x)
        
        x = self.block6(x)
        middle_output6 = self.aux_mlp6(x)
        
        x = self.block7(x)
        middle_output7 = self.aux_mlp7(x)
        
        x = self.block8(x)
        middle_output8 = self.aux_mlp8(x)
        
        x = self.block9(x)
        middle_output9 = self.aux_mlp9(x)
        
        x = self.block10(x)
        middle_output10 = self.aux_mlp10(x)
        
        x = self.block11(x)
        middle_output11 = self.aux_mlp11(x)
        
        x = self.block12(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]  # just tje CLS token
        x = self.head(cls_token_final)

        return x, middle_output1, middle_output2, middle_output3, middle_output4, middle_output5, middle_output6, middle_output7, middle_output8, middle_output9, middle_output10, middle_output11