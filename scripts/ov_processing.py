from PIL import Image, ImageOps
import tifffile as tif
import numpy as np
import os
import json
import cv2
import csv

# 完全禁用 Pillow 的像素限制
Image.MAX_IMAGE_PIXELS = None

def extract_contours(image_path):
    try:
        img = Image.open(image_path).convert('RGB')  # 使用Pillow读取图像并转换为RGB模式
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    except OSError as e:
        print(f"Error extracting contours from {image_path}: {e}")
        return None

def change_background(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        contours = extract_contours(image_path)
        if contours is None:  # 如果提取轮廓失败，直接返回 None
            return None
        new_img_np = np.ones_like(img_np) * 255
        cv2.drawContours(new_img_np, contours, -1, (0, 0, 0), thickness=cv2.FILLED)
        return Image.fromarray(new_img_np.astype('uint8'))
    except OSError as e:
        print(f"Error changing background for {image_path}: {e}")
        return None

def combine_images(image_path1, image_path2):
    try:
        img1 = Image.open(image_path1).convert('RGB')
        img2 = Image.open(image_path2).convert('RGB')
        img1_np = np.array(img1)
        img2_np = np.array(img2)
        contours = extract_contours(image_path1)
        if contours is None:  # 如果提取轮廓失败，直接返回 None
            return None
        new_img_np = np.ones_like(img2_np) * 255
        cv2.drawContours(new_img_np, contours, -1, (0, 0, 0), thickness=cv2.FILLED)
        result = np.where((new_img_np == [0, 0, 0]).all(axis=2, keepdims=True), img1_np, img2_np)
        return Image.fromarray(result.astype('uint8'))
    except OSError as e:
        print(f"Error combining images {image_path1} and {image_path2}: {e}")
        return None

def convert_png_to_tiled_tiff(input_path, output_dir, tile_size=(128, 128), compression=8, predictor=2):
    """将 PNG 文件转换为单分辨率、瓦片存储的 TIFF 文件，并保持原始尺寸"""
    try:
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 获取输入文件名（不带扩展名）
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        # 构建输出文件路径
        output_path = os.path.join(output_dir, f"{base_name}.tif")
        # 读取原始 PNG 文件
        image = Image.open(input_path)
        image_np = np.array(image)
        # 如果图像是 RGB 格式，确保它是 3 通道
        if len(image_np.shape) == 2:  # 如果是灰度图像，转换为 RGB
            image_np = np.stack([image_np] * 3, axis=-1)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 4:  # 如果是 RGBA，去掉 Alpha 通道
            image_np = image_np[:, :, :3]
        # 准备元数据
        metadata = {
            'Software': '',
            'Description': '',
            'shape': list(image_np.shape)  # 将图像形状信息添加到元数据中
        }
        # 将元数据转换为 JSON 字符串
        description = json.dumps(metadata)
        # 将图像保存为瓦片存储的 TIFF 文件
        with tif.TiffWriter(output_path, bigtiff=True) as tif_writer:
            tif_writer.write(
                image_np,
                photometric='rgb',
                compression=compression,
                resolution=(10, 1),  # 设置分辨率为 10 DPI
                description=description,  # 使用 JSON 格式的元数据
                tile=tile_size,
                predictor=predictor,
                planarconfig='contig'  # 使用连续平面配置
            )
        print(f"Image converted and saved to {output_path}")
    except OSError as e:
        print(f"Error converting image to TIFF {input_path}: {e}")

def has_been_processed(png_file, output_dir):
    """检查对应于给定PNG文件的TIFF文件是否已存在于输出目录中"""
    base_name = os.path.splitext(os.path.basename(png_file))[0]
    tif_file = os.path.join(output_dir, f"{base_name}.tif")
    return os.path.exists(tif_file)

if __name__ == "__main__":
    input_image_dir = "/public/home/wang/share_group_folder_wang/pathology/ov_images/raw_datasets/ubc_ocean/train_images"
    mask_output_dir = "/public/home/wang/liujx/prov-gigapath-main/11111ovarian/mask"
    combined_output_dir = "/public/home/wang/liujx/prov-gigapath-main/11111ovarian/combined_output"
    final_tif_output_dir = "/public/home/wang/liujx/prov-gigapath-main/11111ovarian/finaltif"
    failed_csv_path = "/public/home/wang/liujx/prov-gigapath-main/11111ovarian/fail.csv"

    # 确保输出文件夹存在
    os.makedirs(mask_output_dir, exist_ok=True)
    os.makedirs(combined_output_dir, exist_ok=True)
    os.makedirs(final_tif_output_dir, exist_ok=True)

    # 准备记录失败的文件
    failed_files = []

    # 遍历输入文件夹中的所有PNG图片
    for file_name in os.listdir(input_image_dir):
        if file_name.endswith('.png'):
            png_file = os.path.join(input_image_dir, file_name)
            # 检查是否已经处理过
            if has_been_processed(png_file, final_tif_output_dir):
                print(f"File {file_name} has already been processed, skipping.")
                continue

            try:
                # 生成mask图像并保存
                mask_img = change_background(png_file)
                if mask_img is None:
                    print(f"Failed to generate mask for {file_name}, skipping.")
                    failed_files.append((file_name, "Failed to generate mask"))
                    continue

                mask_file_name = os.path.join(mask_output_dir, file_name.replace('.png', '_mask.png'))
                mask_img.save(mask_file_name)  # 使用PIL的save方法保存图像

                # 合并图像并保存
                combined_img = combine_images(png_file, mask_file_name)
                if combined_img is None:
                    print(f"Failed to combine images for {file_name}, skipping.")
                    failed_files.append((file_name, "Failed to combine images"))
                    continue

                combined_file_name = os.path.join(combined_output_dir, file_name)
                combined_img.save(combined_file_name)

                # 将合并后的图像转换为TIFF文件并保存
                convert_png_to_tiled_tiff(combined_file_name, final_tif_output_dir)

            except Exception as e:
                print(f"An error occurred while processing {file_name}: {e}")
                failed_files.append((file_name, str(e)))

    # 写入失败的文件信息到CSV文件
    with open(failed_csv_path, mode='w', newline='') as failed_csv:
        writer = csv.writer(failed_csv)
        writer.writerow(['File Name', 'Error Message'])  # 写入表头
        writer.writerows(failed_files)  # 写入失败文件信息

    print(f"Processing complete. Failed files have been logged to {failed_csv_path}.")