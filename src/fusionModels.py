from functools import partial

import pretrainedmodels.models as premodels
from torch import nn
import torch.nn.functional as F
from VIT_model import *
import torch


class NLBlockND(nn.Module):
    """
    Non‑Local Block with cross‑attention between two branches.
    (Implementation unchanged, kept for completeness.)
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


class VisionTransformer_base(nn.Module):
    """
    ViT‑7 backbone for BreastDM.
    Defaults: img_size=96, patch_size=16, in_c=9, depth=7.
    """
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

        # Initialize weights
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
    """Weight initialisation for ViT and new CNN layers."""
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


class FCUUp(nn.Module):
    """
    Feature Coupling Unit: Transformer tokens → CNN feature maps.
    inplanes=768 (ViT embed dim), outplanes=512 (to match CNN branch).
    """
    def __init__(self, inplanes=768, outplanes=512, up_stride=2, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6)):
        super(FCUUp, self).__init__()
        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # Remove classification token and reshape to 2D feature map
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))
        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


class FusionM(nn.Module):
    """
    LG‑CAFN model for breast MRI classification.
    Supports multi‑channel input (9 or 17 channels) and ViT‑7 backbone.
    """
    def __init__(self, num_classes=2, in_c=9, load_vit=False):
        super(FusionM, self).__init__()
        self.in_c = in_c
        self.load_vit_flag = load_vit
        self.path = r'./model/vit_base_patch16_224_in21k.pth'

        # ----- ViT branch -----
        # VisionTransformer_base automatically calls _init_vit_weights inside its __init__
        self.vit = VisionTransformer_base(img_size=96, patch_size=16, in_c=in_c,
                                         num_classes=num_classes, depth=7)

        # Optionally load pretrained ViT weights (ignore size mismatches)
        if self.load_vit_flag:
            self._load_pretrained_vit()

        # ----- CNN branch (first two stages of SE‑ResNet50) -----
        model_se = premodels.se_resnet50()
        # Replace the first conv layer to accept `in_c` channels
        old_conv = model_se.layer0[0]
        new_conv = nn.Conv2d(in_c, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding,
                             bias=False)
        # Apply kaiming init to the new conv layer
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out")
        
        # Keep the rest of layer0 unchanged (with pretrained weights)
        self.layer0 = nn.Sequential(new_conv, *model_se.layer0[1:])
        self.layer1 = model_se.layer1
        self.layer2 = model_se.layer2
        # Stage 3 and 4 are not used (as per paper)
        # self.layer3 = model_se.layer3
        # self.layer4 = model_se.layer4

        # ----- Cross‑attention and fusion -----
        self.Nlblock = NLBlockND(in_channels=512)   # operates on 512‑dim feature maps
        self.fcuup = FCUUp(inplanes=768, outplanes=512, up_stride=2)

        # ----- Classifier -----
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)      # 512 (CNN) + 512 (ViT after FCU)

        # Initialize ONLY the new layers (ViT already initialized, CNN loaded from pretrained)
        self._init_new_layers()

    def _init_new_layers(self):
        """Initialize only newly added layers (FCU, NLBlock, classifier)."""
        for m in [self.fcuup, self.Nlblock, self.fc]:
            m.apply(_init_vit_weights)

    def _load_pretrained_vit(self):
        """Load pretrained ViT weights, ignoring mismatched layers (e.g. head, pos_embed)."""
        try:
            state_dict = torch.load(self.path, map_location='cpu')
            
            # Remove classification head if present
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                state_dict.pop(k, None)
            
            # Load with strict=False – mismatched layers keep _init_vit_weights values
            missing_keys, unexpected_keys = self.vit.load_state_dict(state_dict, strict=False)
            
            print(f"✅ Loaded pretrained ViT from {self.path}")
            if missing_keys:
                print(f"   Missing keys (kept from _init_vit_weights): {missing_keys}")
            if unexpected_keys:
                print(f"   Unexpected keys (ignored): {unexpected_keys}")
                
        except FileNotFoundError:
            print(f"⚠️  Pretrained ViT not found at {self.path}. Training from scratch.")
        except Exception as e:
            print(f"⚠️  Error loading pretrained weights: {e}. Training from scratch.")

    def forward(self, x):
        # ViT pathway
        vit_x = self.vit(x)                         # (B, num_patches+1, 768)
        # Compute spatial dimensions from number of patches
        num_patches = vit_x.size(1) - 1             # exclude cls token
        H = W = int(num_patches ** 0.5)            # 6 for 96x96 input
        vit_feat = self.fcuup(vit_x, H, W)          # (B, 512, 12, 12)

        # CNN pathway (first two stages)
        cnn_feat = self.layer0(x)
        cnn_feat = self.layer1(cnn_feat)
        cnn_feat = self.layer2(cnn_feat)            # output: (B, 512, 12, 12) for 96x96 input

        # Cross‑attention fusion
        out1 = self.Nlblock(cnn_feat, vit_feat)     # CNN queries ViT
        out2 = self.Nlblock(vit_feat, cnn_feat)     # ViT queries CNN
        fused = torch.cat((out1, out2), dim=1)      # (B, 1024, 12, 12)

        # Classification
        pooled = self.avgpool(fused)                # (B, 1024, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)
        logits = self.fc(pooled)
        return logits


if __name__ == '__main__':
    # Quick test: 9‑channel input 96x96
    print("Testing with 9 channels (Exp-1)...")
    dummy = torch.rand(2, 9, 96, 96)
    model = FusionM(num_classes=2, in_c=9, load_vit=False)
    out = model(dummy)
    print("Output shape:", out.shape)  # Expected: [2, 2]

    # Test with 17 channels
    print("\nTesting with 17 channels (Exp-2)...")
    dummy17 = torch.rand(2, 17, 96, 96)
    model17 = FusionM(num_classes=2, in_c=17, load_vit=False)
    out17 = model17(dummy17)
    print("Output shape:", out17.shape)  # Expected: [2, 2]
    
    # Test with pretrained weights
    print("\nTesting with pretrained ViT weights...")
    model_pretrained = FusionM(num_classes=2, in_c=17, load_vit=True)
    out_pretrained = model_pretrained(dummy17)
    print("Output shape:", out_pretrained.shape)  # Expected: [2, 2]