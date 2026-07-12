# fusionModels.py - Sửa lỗi CUDA misaligned address và shape mismatch
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import timm
from VIT_model import *  # File VIT_model.py của tác giả (ViT-7)

# -------------------- Non-Local Block (sửa lỗi contiguous) --------------------
class NLBlockND(nn.Module):
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
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
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

        if self.mode in ["embedded", "dot", "concatenate"]:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x_thisBranch, x_otherBranch):
        batch_size = x_thisBranch.size(0)

        # ---- g(x) ----
        g_x = self.g(x_thisBranch).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1).contiguous()

        if self.mode == "gaussian":
            theta_x = x_thisBranch.view(batch_size, self.in_channels, -1)
            phi_x = x_otherBranch.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1).contiguous()
            f = torch.matmul(theta_x, phi_x)
        elif self.mode in ["embedded", "dot"]:
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, -1)
            phi_x = phi_x.permute(0, 2, 1).contiguous()
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

        if self.mode in ["gaussian", "embedded"]:
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

# -------------------- Vision Transformer Base (giữ nguyên, đảm bảo forward trả về sequence) --------------------
class VisionTransformer_base(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=2,
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

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x):
        x = self.patch_embed(x)  # [B, 196, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        # TRẢ VỀ TOÀN BỘ SEQUENCE (chứ không chỉ class token)
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

# -------------------- FCUUp (sửa lỗi contiguous) --------------------
class FCUUp(nn.Module):
    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6)):
        super(FCUUp, self).__init__()
        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # Bỏ cls token (x[:, 1:]) và reshape thành feature map
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W).contiguous()
        x_r = self.act(self.bn(self.conv_project(x_r)))
        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))

# -------------------- FusionM (sửa lỗi shape và contiguous) --------------------
class FusionM(nn.Module):
    def __init__(self, num_classes=2, load_vit=False, vit_pretrained_path=None):
        super(FusionM, self).__init__()
        # CNN branch (SENet50 từ timm)
        model_se = timm.create_model('seresnet50', pretrained=True)
        self.layer0 = nn.Sequential(
            model_se.conv1,
            model_se.bn1,
            model_se.act1,
            model_se.maxpool
        )
        self.layer1 = model_se.layer1
        self.layer2 = model_se.layer2
        self.layer3 = model_se.layer3  # giữ nhưng không dùng

        # ViT branch
        self.vit = VisionTransformer_base()  # depth=7
        self.fcuup = FCUUp(inplanes=768, outplanes=512, up_stride=2)

        self.Nlblock = NLBlockND(in_channels=512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

        # Biến tương thích code cũ
        self.load_true = load_vit
        self.path = vit_pretrained_path

        # Nếu có đường dẫn pretrained ViT, load vào
        if vit_pretrained_path is not None:
            self.load_vit_weights_from_12(vit_pretrained_path)
        elif load_vit:
            # Nếu load_vit=True nhưng không có đường dẫn, thử load từ self.path (cũ)
            if hasattr(self, 'path') and self.path is not None:
                self.load_vit_weights_from_12(self.path)
            else:
                print("Warning: load_vit=True but no pretrained path provided. ViT will be randomly initialized.")

    def load_vit_weights_from_12(self, vit12_weights_path):
        """
        Load pretrained ViT-12 weights, trích xuất 7 block đầu, 
        và gán vào self.vit (ViT-7).
        """
        print(f"Loading ViT-12 pretrained weights from {vit12_weights_path} ...")
        # 1. Load pretrained ViT-12 state_dict
        vit12_dict = torch.load(vit12_weights_path, map_location='cpu')
        
        # 2. Lấy state_dict của ViT-7 (self.vit)
        vit7_dict = self.vit.state_dict()
        
        # 3. Lọc các key của 7 block đầu (blocks.0 đến blocks.6)
        # và các key khác (patch_embed, cls_token, pos_embed, norm)
        new_dict = {}
        for k, v in vit12_dict.items():
            # Bỏ qua head (không dùng)
            if k.startswith('head'):
                continue
            # Chỉ lấy các key thuộc 7 block đầu
            if k.startswith('blocks.'):
                block_idx = int(k.split('.')[1])
                if block_idx >= 7:
                    continue
            # Chỉ lấy các key có trong vit7_dict và shape khớp
            if k in vit7_dict and v.shape == vit7_dict[k].shape:
                new_dict[k] = v
            elif k in vit7_dict and v.shape != vit7_dict[k].shape:
                print(f"Shape mismatch for {k}: pretrained {v.shape} vs model {vit7_dict[k].shape}. Skipping.")
        
        # 4. Load vào self.vit (cho phép thiếu key)
        self.vit.load_state_dict(new_dict, strict=False)
        print("Loaded ViT-7 weights extracted from ViT-12 pretrained model.")
        
        # Kiểm tra số lượng key đã load
        loaded_keys = set(new_dict.keys())
        total_keys = set(vit7_dict.keys())
        missing = total_keys - loaded_keys
        if missing:
            print(f"Missing keys in ViT-7 after loading: {missing}")

    def forward(self, x):
        # ---- ViT branch ----
        vit_x = self.vit(x)                      # (B, 197, 768)
        vit_x = self.fcuup(vit_x, 14, 14)        # (B, 512, 28, 28)
        vit_x = F.interpolate(vit_x, size=(14, 14), mode='bilinear', align_corners=False)  # (B, 512, 14, 14)

        # ---- CNN branch ----
        x = self.layer0(x)    # (B, 64, 112, 112)
        x = self.layer1(x)    # (B, 256, 56, 56)
        se_x = self.layer2(x) # (B, 512, 28, 28)
        se_x = F.interpolate(se_x, size=(14, 14), mode='bilinear', align_corners=False)  # (B, 512, 14, 14)

        # ---- Cross-attention fusion ----
        x_path1 = self.Nlblock(se_x, vit_x)   # (B, 512, 14, 14)
        x_path2 = self.Nlblock(vit_x, se_x)   # (B, 512, 14, 14)
        out = torch.cat([x_path1, x_path2], dim=1)  # (B, 1024, 14, 14)

        # ---- Global pooling và classification ----
        out = self.avgpool(out)                # (B, 1024, 1, 1)
        out = out.view(out.size(0), -1)        # (B, 1024)
        out = self.fc(out)                     # (B, num_classes)
        return out

if __name__ == '__main__':
    # Kiểm tra nhanh
    a = torch.rand(2, 3, 224, 224)
    model = FusionM(num_classes=2, load_vit=False)
    out = model(a)
    print(out.shape)