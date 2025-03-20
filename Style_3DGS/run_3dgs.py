import argparse

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from Style_3DGS.render import run_3dgs_rendering
from Style_3DGS.train import run_3dgs_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run localized style transfer with background segmentation."
    )

    parser.add_argument(
        "--content", type=str, required=True, help="Path to the content directory containing original model views."
    )
    parser.add_argument(
        "--style", type=str, required=True, help="Path to the style image."
    )
    parser.add_argument(
        "--output", type=str, default="output", help="Output directory."
    )
    parser.add_argument(
        "--use_depth", action="store_true", help="Enable depth-aware stylization."
    )

    args = parser.parse_args()

    run_3dgs_training(
        source_path=args.content,
        style_image=args.style,
        output_folder=args.output,
        use_depth=args.use_depth,
        iterations=15000,
        freeze_iters=7000,
        depth_offset=0.5,
        depth_prominence=20,
    )

    run_3dgs_rendering(model_path=args.output, style_image=args.style)
