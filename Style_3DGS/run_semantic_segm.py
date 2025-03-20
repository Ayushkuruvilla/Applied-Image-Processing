import argparse

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from Style_3DGS.localized_style_transfer import run_localized_style_transfer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run localized style transfer with background segmentation."
    )

    parser.add_argument(
        "--content", type=str, required=True, help="Path to the content image."
    )
    parser.add_argument(
        "--style", type=str, required=True, help="Path to the style image."
    )
    parser.add_argument(
        "--output", type=str, default="output", help="Output directory."
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="stylized",
        help="Output file name without extension.",
    )
    parser.add_argument(
        "--use_depth", action="store_true", help="Enable depth-aware stylization."
    )

    args = parser.parse_args()

    output_path = run_localized_style_transfer(
        content_img_path=args.content,
        style_img_path=args.style,
        output_path=args.output,
        file_name=args.file_name,
        use_depth=args.use_depth,
    )
