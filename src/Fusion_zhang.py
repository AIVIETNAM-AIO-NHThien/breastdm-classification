from functools import partial
import math

import pretrainedmodels
from torch import nn
import torch.nn.functional as F
from VIT_model import *          # VisionTransformer_base, PatchEmbed, Block, ...
import torch


# ============================================================
# Helper: Zhang-style channel inflation
# ============================================================
def zhang_style_inflate(weight_2d, in_c_new):
    """
    Chuyển trọng số conv 2D từ 3 kênh RGB sang in_c_new kênh
    theo phương pháp Zhang et al. (2022):

        W_new[c] = (W_R + W_G + W_B) / in_c_new,  với mọi c

    Bảo toàn tổng trọng số trên trục kênh:
        sum_c W_new[c] = W_R + W_G + W_B

    Args:
        weight_2d: tensor (out_c, 3, kH, kW) — trọng số pretrained
        in_c_new:  số kênh đầu vào mới (>= 1)

    Returns:
        tensor (out_c, in_c_new, kH, kW)

    Note:
        Nếu in_c_new == 3, hàm vẫn chạy nhưng sẽ làm mất sự khác biệt
        giữa 3 kênh RGB. Với DCE-MRI (in_c=9) thì không ảnh hưởng.
    """
    assert weight_2d.dim() == 4 and weight_2d.size(1) == 3, \
        f"Expected weight shape (out_c, 3, kH, kW), got {tuple(weight_2d.shape)}"

    # 1. Tổng 3 kênh RGB: (out_c, 1, kH, kW)
    sum_w = weight_2d.sum(dim=1, keepdim=True)
    # 2. Average inflation: chia N
    mean_w = sum_w / float(in_c_new)
    # 3. Expand sang N kênh
    return mean_w.expand(-1, in_c_new, -1, -1).contiguous()


# ============================================================
# Non-local block (giữ nguyên)
# ============================================================
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
            theta_x = x_otherBranch.view(batch_size, self.in_channels, -1)
            phi_x = x_thisBranch.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)
        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x_otherBranch).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x_thisBranch).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)
        else:  # concatenate
            theta_x = self.theta(x_otherBranch).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x_thisBranch).view(batch_size, self.inter_channels, 1, -1)
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


# ============================================================
# Vision Transformer base (giữ nguyên)
# ============================================================
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


# ============================================================
# FCUUp (giữ nguyên)
# ============================================================
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
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))
        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


# ============================================================
# FusionM — với Zhang-style initialization
# ============================================================
class FusionM(nn.Module):
    def __init__(self, num_classes=2, in_c=9, load_vit=False, vit_path=None,
                 img_size=96, patch_size=16):
        super(FusionM, self).__init__()

        # ==================== SE-ResNet50 branch ====================
        model_se = pretrainedmodels.se_resnet50(pretrained='imagenet')

        old_conv = model_se.layer0[0]   # Conv2d(3, 64, k=7, s=2, p=3, bias=False)
        new_conv = nn.Conv2d(
            in_c, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            # ---------- Zhang-style channel inflation ----------
            # W_new[c] = (W_R + W_G + W_B) / in_c,  với mọi c
            # Bảo toàn tổng trọng số trên trục kênh (scale-preserving)
            inflated = zhang_style_inflate(old_conv.weight, in_c)   # (64, in_c, 7, 7)
            new_conv.weight.copy_(inflated)

            # Ghi log để debug
            print(f"[SE-ResNet50] conv1: {tuple(old_conv.weight.shape)} "
                  f"→ {tuple(new_conv.weight.shape)} (Zhang-style)")
            # Kiểm tra: tất cả kênh giống hệt nhau
            assert torch.allclose(new_conv.weight[:, 0], new_conv.weight[:, -1], atol=1e-7), \
                "Zhang-style: các kênh phải giống nhau"
            # Kiểm tra: tổng trên trục kênh được bảo toàn
            assert torch.allclose(
                new_conv.weight.sum(dim=1),
                old_conv.weight.sum(dim=1),
                atol=1e-6,
            ), "Zhang-style: tổng trọng số trên trục kênh bị lệch"

        model_se.conv1 = new_conv

        self.layer0 = nn.Sequential(
            new_conv,               # conv1
            model_se.layer0[1],     # bn1
            model_se.layer0[2],     # relu
            model_se.layer0[3],     # maxpool
        )
        self.layer1 = model_se.layer1
        self.layer2 = model_se.layer2

        # ==================== ViT branch ====================
        self.vit = VisionTransformer_base(
            img_size=img_size, patch_size=patch_size, in_c=in_c
        )
        for param in self.vit.parameters():
            param.requires_grad = True

        # ==================== Non-local + Fusion ====================
        self.Nlblock = NLBlockND(in_channels=512)
        self.fcuup = FCUUp(inplanes=768, outplanes=512, up_stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(1024, num_classes)

        # Load pretrained ViT (Zhang-style)
        if load_vit and vit_path is not None:
            self._load_pretrained_vit(vit_path)
        elif load_vit:
            print("Warning: load_vit=True nhưng vit_path không được chỉ định. "
                  "Bỏ qua load pretrained ViT.")

    # ------------------------------------------------------------
    def _load_pretrained_vit(self, vit_path):
        """Load pretrained ViT với Zhang-style channel inflation
        và nội suy position embedding."""
        weight_dict = torch.load(vit_path, map_location='cpu')

        # Bỏ head gốc
        for k in ['head.weight', 'head.bias']:
            if k in weight_dict:
                del weight_dict[k]

        # ========== 1. Patch embedding: 3 kênh → in_c kênh (Zhang-style) ==========
        if 'patch_embed.proj.weight' in weight_dict:
            old_weight = weight_dict['patch_embed.proj.weight']   # (768, 3, 16, 16)
            in_c = self.vit.patch_embed.proj.in_channels
            new_weight = zhang_style_inflate(old_weight, in_c)    # (768, in_c, 16, 16)

            weight_dict['patch_embed.proj.weight'] = new_weight
            print(f"[ViT] patch_embed.proj: {tuple(old_weight.shape)} "
                  f"→ {tuple(new_weight.shape)} (Zhang-style)")

            # Kiểm tra bảo toàn scale
            assert torch.allclose(
                new_weight.sum(dim=1), old_weight.sum(dim=1), atol=1e-6
            ), "Zhang-style ViT: tổng trọng số bị lệch"

        # patch_embed.proj.bias (nếu có) — không đổi vì không phụ thuộc in_c

        # ========== 2. Position embedding: nội suy về kích thước ảnh hiện tại ==========
        if 'pos_embed' in weight_dict:
            pos_embed_pretrained = weight_dict['pos_embed']
            cls_token_pos = pos_embed_pretrained[:, :1, :]
            pos_embed_patches = pos_embed_pretrained[:, 1:, :]

            num_patches_pret = pos_embed_patches.shape[1]
            grid_size_pret = int(math.sqrt(num_patches_pret))
            pos_embed_patches = pos_embed_patches.reshape(
                1, grid_size_pret, grid_size_pret, -1
            ).permute(0, 3, 1, 2)

            grid_size_new = self.vit.patch_embed.grid_size[0]
            pos_embed_patches_new = F.interpolate(
                pos_embed_patches, size=(grid_size_new, grid_size_new),
                mode='bicubic', align_corners=False,
            )
            pos_embed_patches_new = pos_embed_patches_new.permute(0, 2, 3, 1).reshape(
                1, -1, 768
            )

            new_pos_embed = torch.cat([cls_token_pos, pos_embed_patches_new], dim=1)
            weight_dict['pos_embed'] = new_pos_embed
            print(f"[ViT] pos_embed: {tuple(pos_embed_pretrained.shape)} "
                  f"→ {tuple(new_pos_embed.shape)} (bicubic interp)")

        # ========== 3. Load state_dict ==========
        missing, unexpected = self.vit.load_state_dict(weight_dict, strict=False)
        if missing:
            print(f"[ViT] Missing keys (khởi tạo mới): {missing[:5]}"
                  f"{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[ViT] Unexpected keys: {unexpected[:5]}"
                  f"{'...' if len(unexpected) > 5 else ''}")
        print(f"✅ Loaded pretrained ViT (Zhang-style) from {vit_path}")

    # ------------------------------------------------------------
    def forward(self, x):
        # ViT forward
        vit_x = self.vit(x)                                   # (B, num_tokens+1, 768)
        H_grid, W_grid = self.vit.patch_embed.grid_size       # 6, 6
        vit_x = self.fcuup(vit_x, H_grid, W_grid)             # (B, 512, 12, 12)

        # CNN forward
        x = self.layer0(x)
        x = self.layer1(x)
        se_x = self.layer2(x)                                 # (B, 512, 12, 12)

        # Cross-attention fusion
        x_path1 = self.Nlblock(se_x, vit_x)                   # (B, 512, 12, 12)
        x_path2 = self.Nlblock(vit_x, se_x)                   # (B, 512, 12, 12)
        out = torch.cat((x_path1, x_path2), 1)                # (B, 1024, 12, 12)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ============================================================
# Test
# ============================================================
if __name__ == '__main__':
    # ---- Test 1: Zhang-style helper ----
    print("\n=== Test 1: zhang_style_inflate ===")
    torch.manual_seed(0)
    old_w = torch.randn(64, 3, 7, 7)
    new_w = zhang_style_inflate(old_w, 9)
    assert new_w.shape == (64, 9, 7, 7)
    for c in range(1, 9):
        assert torch.allclose(new_w[:, 0], new_w[:, c], atol=1e-7)
    assert torch.allclose(new_w.sum(dim=1), old_w.sum(dim=1), atol=1e-6)

    # Test tính tương đương output với ảnh xám
    x_gray = torch.randn(1, 1, 32, 32)
    x_9ch = x_gray.expand(-1, 9, -1, -1)
    x_rgb = x_gray.expand(-1, 3, -1, -1)
    y_rgb = F.conv2d(x_rgb, old_w, stride=1, padding=3)
    y_9ch = F.conv2d(x_9ch, new_w, stride=1, padding=3)
    assert torch.allclose(y_rgb, y_9ch, atol=1e-5), \
        f"Max diff: {(y_rgb - y_9ch).abs().max()}"
    print("✓ zhang_style_inflate PASS")

    # ---- Test 2: FusionM forward (không load pretrained ViT) ----
    print("\n=== Test 2: FusionM forward ===")
    a = torch.rand(2, 9, 96, 96)
    model = FusionM(num_classes=2, in_c=9, load_vit=False)
    out = model(a)
    print(f"Input: {tuple(a.shape)} → Output: {tuple(out.shape)}")
    assert out.shape == (2, 2)
    print("✓ FusionM forward PASS")

    # ---- Test 3: kiểm tra conv1 trong FusionM đã được Zhang-style ----
    print("\n=== Test 3: kiểm tra conv1 trong FusionM ===")
    conv1_w = model.layer0[0].weight
    print(f"conv1.weight shape: {tuple(conv1_w.shape)}")
    # Tất cả kênh phải giống hệt nhau
    for c in range(1, 9):
        assert torch.allclose(conv1_w[:, 0], conv1_w[:, c], atol=1e-7), \
            f"Kênh {c} khác kênh 0"
    print("✓ Tất cả 9 kênh của conv1 giống hệt nhau (Zhang-style)")
    print("\n🎉 Tất cả test PASS")