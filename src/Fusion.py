from functools import partial
from collections import OrderedDict

import pretrainedmodels.models as premodels
from torch import nn
import torch.nn.functional as F
import torch


class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=2, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions"""
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x_thisBranch, x_otherBranch):
        batch_size = x_thisBranch.size(0)

        g_x = self.g(x_thisBranch).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x_thisBranch.view(batch_size, self.in_channels, -1)
            phi_x = x_otherBranch.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, -1)
            phi_x = phi_x.permute(0, 2, 1)
            f = torch.matmul(phi_x, theta_x)

        else:  # concatenate
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        else:
            N = f.size(-1)
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x_thisBranch.size()[2:])

        W_y = self.W_z(y)
        z = W_y + x_thisBranch
        return z


# ==================== PATCH EMBEDDING (HỖ TRỢ 9 CHANNELS) ====================
class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# ==================== BLOCK COMPONENTS ====================
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ==================== VISION TRANSFORMER CHO 9 CHANNELS ====================
class VisionTransformer_9ch(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=9, num_classes=2,
                 embed_dim=768, depth=7, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        super(VisionTransformer_9ch, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features(self, x):
        """Extract features before classification head"""
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x  # Return all tokens (B, N+1, embed_dim)

    def forward(self, x):
        x = self.forward_features(x)
        if self.dist_token is None:
            x = self.pre_logits(x[:, 0])  # Take cls_token
        else:
            x = x[:, 0], x[:, 1]
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


# ==================== SE-RESNET CHO 9 CHANNELS ====================
class SEResNet_9ch(nn.Module):
    """SE-ResNet50 modified for 9-channel input"""
    def __init__(self):
        super().__init__()
        model_se = premodels.se_resnet50()
        
        # Sửa conv layer đầu từ 3 → 9 channels
        old_conv = model_se.layer0.conv1
        new_conv = nn.Conv2d(
            in_channels=9,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Copy weights từ pretrained (3 channels) sang 9 channels
        with torch.no_grad():
            weight_3ch = old_conv.weight.data  # (64, 3, 7, 7)
            # Lặp lại 3 lần để có 9 channels
            weight_9ch = torch.cat([weight_3ch, weight_3ch, weight_3ch], dim=1)
            new_conv.weight.data = weight_9ch
        
        model_se.layer0.conv1 = new_conv
        
        self.layer0 = model_se.layer0
        self.layer1 = model_se.layer1
        self.layer2 = model_se.layer2
    
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)  # (B, 512, 28, 28)
        return x


# ==================== FCUUP (GIỮ NGUYÊN) ====================
class FCUUp(nn.Module):
    """Transformer patch embeddings -> CNN feature maps"""
    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6)):
        super(FCUUp, self).__init__()
        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # x shape: (B, N+1, C) -> (B, C, H, W)
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))
        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


# ==================== FUSION MODEL HOÀN CHỈNH ====================
class FusionM_9ch(nn.Module):
    def __init__(self, num_classes=2, load_vit=False, vit_pretrained_path=None,
                 drop_ratio=0.0, attn_drop_ratio=0.0):
        super(FusionM_9ch, self).__init__()
        
        # CNN branch with 9 channels
        self.cnn = SEResNet_9ch()
        
        # ViT branch with 9 channels (có thể điều chỉnh dropout)
        self.vit = VisionTransformer_9ch(
            img_size=224,
            patch_size=16,
            in_c=9,                 # 9 channels
            num_classes=num_classes,
            embed_dim=768,
            depth=12,
            num_heads=12,
            drop_ratio=drop_ratio,          # ← tham số từ config
            attn_drop_ratio=attn_drop_ratio # ← tham số từ config
        )
        
        # Non-local block
        self.nl_block = NLBlockND(in_channels=512, dimension=2)
        
        # FCUUp để chuyển ViT features sang spatial
        self.fcuup = FCUUp(inplanes=768, outplanes=512, up_stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)  # 512 + 512 = 1024
        
        # Load pretrained ViT if specified
        self.load_vit = load_vit
        if load_vit and vit_pretrained_path:
            self._load_vit_pretrained(vit_pretrained_path)
    
    def _load_vit_pretrained(self, path):
        """Load pretrained ViT weights (3 channels) and adapt to 9 channels"""
        print(f"Loading pretrained ViT from {path}")
        pretrained_dict = torch.load(path, map_location='cpu')
        
        # Remove classification head (we use our own)
        keys_to_remove = ['head.weight', 'head.bias']
        if 'head_dist.weight' in pretrained_dict:
            keys_to_remove.extend(['head_dist.weight', 'head_dist.bias'])
        
        for k in keys_to_remove:
            if k in pretrained_dict:
                del pretrained_dict[k]
        
        # Adapt patch embedding from 3 to 9 channels
        if 'patch_embed.proj.weight' in pretrained_dict:
            old_weight = pretrained_dict['patch_embed.proj.weight']  # (768, 3, 16, 16)
            # Repeat 3 times to get 9 channels
            new_weight = old_weight.repeat(1, 3, 1, 1)  # (768, 9, 16, 16)
            pretrained_dict['patch_embed.proj.weight'] = new_weight
            print(f"Adapted patch embedding from 3 to 9 channels")
        
        # Load weights (strict=False to ignore mismatched keys)
        missing_keys, unexpected_keys = self.vit.load_state_dict(pretrained_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:5]}...")
        print("Pretrained ViT loaded successfully")
    
    def forward(self, x):
        """
        Args:
            x: (B, 9, 224, 224) - 9 channels (1 pre + 8 post-contrast)
        Returns:
            output: (B, num_classes)
        """
        # CNN branch
        cnn_features = self.cnn(x)  # (B, 512, 28, 28)
        
        # ViT branch - get features before classification head
        vit_tokens = self.vit.forward_features(x)  # (B, 197, 768)
        # Convert tokens to spatial feature map
        vit_features = self.fcuup(vit_tokens, 14, 14)  # (B, 512, 28, 28)
        
        cnn_features = F.normalize(cnn_features, dim=1)
        vit_features = F.normalize(vit_features, dim=1)
        # Cross-attention fusion
        fusion_1 = self.nl_block(cnn_features, vit_features)  # CNN query, ViT key/value
        fusion_2 = self.nl_block(vit_features, cnn_features)  # ViT query, CNN key/value
        
        # Concatenate both fusion paths
        combined = torch.cat([fusion_1, fusion_2], dim=1)  # (B, 1024, 28, 28)
        
        # Global pooling and classification
        pooled = self.avgpool(combined)  # (B, 1024, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, 1024)
        output = self.fc(pooled)  # (B, num_classes)
        
        return output


# ==================== MAIN TEST ====================
if __name__ == '__main__':
    # Test với 9 channels
    a = torch.rand(2, 9, 224, 224)
    
    # Khởi tạo model
    model = FusionM_9ch(num_classes=2, load_vit=False)
    
    # Forward pass
    out = model(a)
    print(f"Input shape: {a.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output: {out}")