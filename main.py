from PIL import Image
from matplotlib import pyplot as plt

from Style_3DGS.AdaIN import adain_inference
from Style_3DGS import run_localized_style_transfer

# to test out individual pipelines without GUI
if __name__ == "__main__":
    depth_prominence_values = [1, 3, 5, 10, 100]
    proximity_offset_values = [0, 0.3, 0.5, 0.7, 1]
    image_paths = []

    # for depth in depth_prominence_values:
    #     output_path = adain_inference(
    #         content_img="input/content/brad_pitt.jpg",
    #         style_img="input/style/photo.png",
    #         file_name=f'brad_pitt_{depth}_0',
    #         depth_prominence=depth,
    #         depth_offset=0,
    #         use_depth=True,
    #     )
    #     image_paths.append(output_path)

    for offset in proximity_offset_values:
        output_path = adain_inference(
            content_img="input/content/brad_pitt.jpg",
            style_img="input/style/photo.png",
            file_name=f'brad_pitt_20_{offset}',
            depth_prominence=20,
            depth_offset=offset,
            use_depth=True,
        )
        image_paths.append(output_path)

    fig, axes = plt.subplots(1, len(image_paths), figsize=(20, 5))
    for i, path in enumerate(image_paths):
        img = Image.open(path)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"depth offset: {depth_prominence_values[i]}")

    plt.tight_layout()
    plt.savefig('output/depth_values_comparison.png')
    plt.show()

    # content_img = "input/content/tabby-cat.jpg"
    # style_img = "input/style/gogh.jpg"
    # stylized_image = run_localized_style_transfer(content_img, style_img, output_path='output')

    # img = Image.open(stylized_image)
    # plt.imshow(img)
    # plt.title("Localized semantic segmentation")
    # plt.show()
