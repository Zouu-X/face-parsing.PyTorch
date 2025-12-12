from datasets import load_dataset
from PIL import Image
import numpy as np




def mask_extraction():
    return 0

# Load
def process_batch():
    masks = []
    # "image" 跟着dataset走
    for img in examples["image"]:
        extracted_mask = mask_extraction(img)
        masks.append(extracted_mask)

    return {"mask": masks}

# Main process
if __name__ == "__main__":
    print("加载数据集ing")
    dataset = load_dataset("zhengchong/MakeUpAnyone", split="train")
    print("数据集结构:", dataset)
    print("第一条数据示例:", dataset[0])