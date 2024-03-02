import matplotlib.pyplot as plt
import os
from os.path import join
images = "datalake_folder/image"
clothes = "datalake_folder/cloth"
results = "postprocessing/no_gray"
processed = "postprocessing/final"
out = "report/TestTryOn_MENBlazer_13FEB"

for image in os.listdir(images):
    for cloth in os.listdir(clothes):
        result = f'{image.split(".")[0]}_{cloth.split(".")[0]}.jpg'
        img_ori = plt.imread(join(images,image))
        cloth = plt.imread(join(clothes,cloth))
        img_new = plt.imread(
            join(results,result), 0
        )
        img_new_processed = plt.imread(
            join(processed,result), 0
        )

        plt.figure(figsize=(10, 8))
        grid = plt.GridSpec(2, 5, wspace=0, hspace=0.2)

        plt.subplot(grid[0, 0])
        plt.imshow(img_ori)
        plt.axis("off")
        plt.title("Original")

        plt.subplot(grid[1, 0])
        plt.imshow(cloth)
        plt.axis("off")
        plt.title("Cloth")

        plt.subplot(grid[:, 1:3])
        plt.imshow(img_new)
        plt.axis("off")
        plt.title("Generated")

        plt.subplot(grid[:, 3:5])
        plt.imshow(img_new_processed)
        plt.axis("off")
        plt.title("Generated")

        plt.savefig(join(out,result))