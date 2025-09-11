import os, argparse, torch
from sam2.build_sam import build_sam2

"""
使用说明:
1. 默认用已存在的 Hiera-L 配置验证脚手架是否可运行。
2. 如果你后来创建了 ViT 配置 (例如 configs/sam2/sam2_vit_dino_l16.yaml)，运行时加:
   python tools/check_vit_neck.py --config configs/sam2/sam2_vit_dino_l16.yaml
3. 仅此脚本被修改；不改动其他源码。
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/sam2/sam2_hiera_l.yaml",
                    help="Hydra 配置名 (已存在的配置，如 Hiera-L；或你新增的 ViT 配置)")
    ap.add_argument("--ckpt", type=str, default=None, help="可选 checkpoint 路径")
    ap.add_argument("--size", type=int, default=640, help="测试输入尺寸 (需为16的倍数)")
    args = ap.parse_args()

    cfg_path = args.config
    # 这里只做存在性提示；Hydra 会在其搜索路径下解析 configs/sam2/... 
    local_fs_path = os.path.join(os.getcwd(), cfg_path)
    if not os.path.exists(local_fs_path) and not cfg_path.endswith("hiera_l.yaml"):
        print(f"[WARN] 本地未找到 {local_fs_path} (如果这是打包内置路径可忽略; 若你期望的是新建的 ViT YAML, 请先创建)")

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
