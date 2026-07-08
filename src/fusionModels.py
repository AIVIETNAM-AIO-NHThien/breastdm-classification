import os
import ssl
from functools import partial

import timm
import torch
from torch import nn
import torch.nn.functional as F

# File này được giả sử chứa các lớp PatchEmbed, Block, OrderedDict (nếu dùng trong VisionTransformer_base)
from VIT_model import *


# ======================== NLBlock (Cross‑Attention) ========================
class NLBlockND(nn.Module):
    """
    Non‑Local Block with cross‑attention between two branches.
    """
    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=2, bn_layer=True):
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

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

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
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x_thisBranch.size()[2:])
        W_y = self.W_z(y)
        z = W_y + x_thisBranch
        return z


# ======================== ViT Base (VisionTransformer_base) ========================
class VisionTransformer_base(nn.Module):
    def __init__(self, img_size=96, patch_size=16, in_c=9, num_classes=2,
                 embed_dim=768, depth=7, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        super(VisionTransformer_base, self).__init__()
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

        # Init weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x


def _init_vit_weights(m):
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


# ======================== FCU (Feature Coupling Unit) ========================
class FCUUp(nn.Module):
    def __init__(self, inplanes=768, outplanes=512, up_stride=2, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6)):
        super(FCUUp, self).__init__()
        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))
        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


# ======================== Main Fusion Model (LG‑CAFN) ========================
class FusionM(nn.Module):
    def __init__(self, num_classes=2, in_c=9, load_vit=False):
        super(FusionM, self).__init__()
        self.in_c = in_c
        self.load_vit_flag = load_vit
        self.path = r'./model/vit_base_patch16_224_in21k.pth'

        # ----- ViT branch -----
        self.vit = VisionTransformer_base(
            img_size=96, patch_size=16, in_c=in_c,
            num_classes=num_classes, depth=7
        )
        if self.load_vit_flag:
            self._load_pretrained_vit()

        # ----- CNN branch (SE‑ResNet50) -----
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        se_resnet = timm.create_model('seresnet50', pretrained=True)

        # Tạo stage0: conv1 + bn1 + act1 + maxpool + layer1 (3 block, stride 4, 256c)
        self.cnn_stage0 = nn.Sequential(
            se_resnet.conv1,
            se_resnet.bn1,
            se_resnet.act1,
            se_resnet.maxpool,
            se_resnet.layer1
        )
        # Stage1: layer2 (4 block, stride 8, 512c) – đây chính là đầu ra ta cần
        self.cnn_stage1 = se_resnet.layer2

        # Thay conv1 để nhận đúng số kênh đầu vào
        old_conv = self.cnn_stage0[0]  # conv1 gốc (3 kênh)
        new_conv = nn.Conv2d(
            in_c, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out")
        self.cnn_stage0[0] = new_conv

        # ----- Fusion -----
        self.Nlblock = NLBlockND(in_channels=512)
        self.fcuup = FCUUp(inplanes=768, outplanes=512, up_stride=2)

        # ----- Classifier -----
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

        self._init_new_layers()

    def _init_new_layers(self):
        for m in [self.fcuup, self.Nlblock, self.fc]:
            m.apply(_init_vit_weights)

    def _load_pretrained_vit(self):
        """Load pretrained ViT, xử lý pos_embed và bỏ qua patch_embed (do khác số kênh)."""
        if not os.path.exists(self.path):
            print(f"⚠️  Pretrained ViT not found at {self.path}")
            return

        state_dict = torch.load(self.path, map_location='cpu')

        # Xoá head không cần thiết
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            state_dict.pop(k, None)

        # Xoá patch_embed (số kênh không khớp, sẽ dùng _init_vit_weights)
        state_dict.pop('patch_embed.proj.weight', None)
        state_dict.pop('patch_embed.proj.bias', None)

        # Nội suy pos_embed
        if 'pos_embed' in state_dict:
            pretrained_pos = state_dict['pos_embed']          # [1, 197, 768]
            current_pos = self.vit.pos_embed                  # [1, 37, 768]
            if pretrained_pos.shape != current_pos.shape:
                print(f"🔄 Interpolating pos_embed: {pretrained_pos.shape} → {current_pos.shape}")
                cls_token = pretrained_pos[:, :1, :]
                patches = pretrained_pos[:, 1:, :]            # [1, 196, 768]
                grid_size = int(patches.shape[1] ** 0.5)      # 14
                new_grid = int((current_pos.shape[1] - 1) ** 0.5)  # 6
                patches = patches.reshape(1, grid_size, grid_size, -1).permute(0, 3, 1, 2)
                patches = F.interpolate(patches, size=(new_grid, new_grid), mode='bicubic')
                patches = patches.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
                state_dict['pos_embed'] = torch.cat([cls_token, patches], dim=1)

        # Load với strict=False – bỏ qua các block thừa (7‑11) và patch_embed bị thiếu
        missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded pretrained ViT (first 7 blocks)")
        if missing:
            print(f"   Missing keys (will be randomly init): {missing}")
        if unexpected:
            print(f"   Unexpected keys (ignored): {len(unexpected)} keys from blocks 7-11")

    def forward(self, x):
        # ViT pathway
        vit_x = self.vit(x)                       # (B, 37, 768)
        num_patches = vit_x.size(1) - 1
        H = W = int(num_patches ** 0.5)           # 6
        vit_feat = self.fcuup(vit_x, H, W)        # (B, 512, 12, 12)

        # CNN pathway: chỉ dùng 2 stage đầu → 512 kênh
        cnn_feat = self.cnn_stage0(x)             # (B, 256, 24, 24)
        cnn_feat = self.cnn_stage1(cnn_feat)      # (B, 512, 12, 12)

        # Cross‑attention
        out1 = self.Nlblock(cnn_feat, vit_feat)   # CNN query ViT
        out2 = self.Nlblock(vit_feat, cnn_feat)   # ViT query CNN
        fused = torch.cat((out1, out2), dim=1)    # (B, 1024, 12, 12)

        # Classifier
        pooled = self.avgpool(fused)
        pooled = pooled.view(pooled.size(0), -1)
        logits = self.fc(pooled)
        return logits

# ======================== Test ========================
if __name__ == '__main__':
    print("Testing with 9 channels (Exp-1)...")
    dummy = torch.rand(2, 9, 96, 96)
    model = FusionM(num_classes=2, in_c=9, load_vit=False)
    out = model(dummy)
    print("Output shape:", out.shape)

    print("\nTesting with 17 channels (Exp-2)...")
    dummy17 = torch.rand(2, 17, 96, 96)
    model17 = FusionM(num_classes=2, in_c=17, load_vit=False)
    out17 = model17(dummy17)
    print("Output shape:", out17.shape)

    print("\nTesting with pretrained ViT weights...")
    model_pretrained = FusionM(num_classes=2, in_c=17, load_vit=True)
    out_pretrained = model_pretrained(dummy17)
    print("Output shape:", out_pretrained.shape)