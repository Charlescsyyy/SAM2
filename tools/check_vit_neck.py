import os, argparse, torch
from sam2.build_sam import build_sam2

"""
使用说明 (统一到 sam2.1 命名空间):
1) 默认使用 sam2.1 的 Hiera-L 配置，快速检查搭建是否正常:
    python tools/check_vit_neck.py
2) 若要检查你新增的 ViT 配置，请传 sam2.1 路径，例如:
    python tools/check_vit_neck.py --config configs/sam2.1/sam2_vit_dino.yaml
    或
    python tools/check_vit_neck.py --config configs/sam2.1/sam2_vit_ijepa.yaml
3) 本脚本只读配置并打印 trunk/neck 维度与 FPN 输出，不改动其他源码。
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Hydra 配置路径 (推荐 sam2.1 命名空间: 如 configs/sam2.1/sam2_vit_dino.yaml)",
    )
    ap.add_argument("--ckpt", type=str, default=None, help="可选 checkpoint 路径")
    ap.add_argument("--size", type=int, default=640, help="测试输入尺寸 (需为16的倍数)")
    args = ap.parse_args()

    cfg_path = args.config
    # 仅做存在性提示；Hydra 会按包内搜索路径解析 configs/... 
    local_fs_path = os.path.join(os.getcwd(), cfg_path)
    if not os.path.exists(local_fs_path) and not cfg_path.endswith("hiera_l.yaml"):
        print(
            f"[WARN] 本地未找到 {local_fs_path}. 如果这是包内置路径可忽略；"
            f"若你期望的是新建的 ViT YAML，请将其放到 configs/sam2.1/ 并传入该路径。"
        )

    model = build_sam2(cfg_path, args.ckpt)
    enc = model.image_encoder

    print(f"Loaded config: {cfg_path}")
    print("channel_list (trunk) =", getattr(enc.trunk, "channel_list", None))
    print("backbone_channel_list (neck) =", getattr(enc.neck, "backbone_channel_list", None))

    H = W = args.size
    assert H % 16 == 0 and W % 16 == 0, "输入尺寸需为16倍数"
    x = torch.zeros(1, 3, H, W)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    model.to(device)

    with torch.inference_mode():
        xs = enc.trunk(x)
        print("Backbone raw levels (len=%d):" % len(xs), [tuple(t.shape) for t in xs])
        fpn_out = enc.neck(xs)
        feats, _pos = fpn_out if isinstance(fpn_out, tuple) else (fpn_out, None)
        if isinstance(feats, (list, tuple)):
            print("FPN levels:", [tuple(t.shape) for t in feats])
        else:
            print("FPN out tensor shape:", tuple(feats.shape))

    print("Done.")

if __name__ == "__main__":
    main()
