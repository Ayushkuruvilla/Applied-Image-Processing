import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from AdaIN.test import adain_inference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stylize an image using AdaIN style transfer."
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
        "--depth_offset",
        type=float,
        default=0.15,
        help="Depth offset for depth-aware style transfer.",
    )
    parser.add_argument(
        "--depth_prominence", type=float, default=20, help="Depth prominence factor."
    )
    parser.add_argument(
        "--use_depth", action="store_true", help="Enable depth-aware stylization."
    )

    args = parser.parse_args()

    adain_inference(
        content_img=args.content,
        style_img=args.style,
        depth_offset=args.depth_offset,
        depth_prominence=args.depth_prominence,
        output=args.output,
        file_name=args.file_name,
        use_depth=args.use_depth,
    )
