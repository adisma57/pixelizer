from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import math
from collections import Counter

def upscale_for_pixel_perfect(img, min_display_size=350):
    """Agrandit l’image avec NEAREST si elle est trop petite, pour affichage pixel-perfect."""
    w, h = img.size
    factor = max(1, min_display_size // max(w, h))
    if factor > 1:
        img = img.resize((w * factor, h * factor), Image.NEAREST)
    return img

def resize_and_contrast(img, width, contrast_value):
    ratio = img.height / img.width
    new_height = int(width * ratio)
    resized_img = img.resize((width, new_height), Image.NEAREST)
    if contrast_value != 1.0:
        enhancer = ImageEnhance.Contrast(resized_img)
        resized_img = enhancer.enhance(contrast_value)
    return resized_img

def find_closest_color(pixel, palette):
    diffs = palette - pixel
    dist = np.sum(diffs**2, axis=1)
    idx = np.argmin(dist)
    return palette[idx]

def draw_block_grid(image, block_size, color=(255, 0, 0), width=1):
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size
    for x in range(block_size, img_w, block_size):
        draw.line([(x, 0), (x, img_h)], fill=color, width=width)
    for y in range(block_size, img_h, block_size):
        draw.line([(0, y), (img_w, y)], fill=color, width=width)
    return image

def split_grid(arr, size):
    h, w = arr.shape[:2]
    blocks = []
    for i in range(0, h, size):
        for j in range(0, w, size):
            blocks.append((i, j, arr[i:i+size, j:j+size]))
    return blocks

def convert_to_gray(resized_img, gray_palette_df, n_grays, rgb_to_code):
    arr = np.array(resized_img)
    if arr.shape[2] == 4:
        rgb_arr = arr[..., :3]
        alpha_arr = arr[..., 3]
    else:
        rgb_arr = arr
        alpha_arr = np.ones(arr.shape[:2], dtype=np.uint8) * 255
    lumin = (0.299 * rgb_arr[..., 0] + 0.587 * rgb_arr[..., 1] + 0.114 * rgb_arr[..., 2]).astype(int)
    gray_rows = gray_palette_df.sort_values("R").reset_index(drop=True)
    gray_steps = np.linspace(0, 255, n_grays)
    grays_palette = []
    for step in gray_steps:
        idx = (np.abs(gray_rows["R"] - step)).idxmin()
        row = gray_rows.loc[idx]
        grays_palette.append([int(row["R"]), int(row["G"]), int(row["B"])])
    grays_palette = np.array(grays_palette)
    gray_img = np.zeros_like(rgb_arr)
    code_grid = np.empty(lumin.shape, dtype=object)
    for i in range(lumin.shape[0]):
        for j in range(lumin.shape[1]):
            if alpha_arr[i, j] == 0:
                code_grid[i, j] = ''
                gray_img[i, j] = [255, 255, 255]
            else:
                idx = np.abs(grays_palette[:, 0] - lumin[i, j]).argmin()
                gray_rgb = grays_palette[idx]
                gray_img[i, j] = gray_rgb
                code_grid[i, j] = rgb_to_code[tuple(gray_rgb)]
    return gray_img, code_grid, alpha_arr

def convert_image_to_palette(resized_img, palette_df, rgb_palette, valid_palette):
    arr = np.array(resized_img)
    if arr.shape[2] == 4:
        rgb_arr = arr[..., :3]
        alpha_arr = arr[..., 3]
    else:
        rgb_arr = arr
        alpha_arr = np.ones(arr.shape[:2], dtype=np.uint8) * 255
    result_img = np.zeros_like(rgb_arr)
    code_grid = np.empty(rgb_arr.shape[:2], dtype=object)
    rgb_to_code = {}
    for i, row in valid_palette.iterrows():
        try:
            rgb_tuple = (int(row['R']), int(row['G']), int(row['B']))
            rgb_to_code[rgb_tuple] = row['Code']
        except:
            continue
    for i in range(rgb_arr.shape[0]):
        for j in range(rgb_arr.shape[1]):
            if alpha_arr[i, j] == 0:
                code_grid[i, j] = ''
                result_img[i, j] = [255, 255, 255]
            else:
                color = find_closest_color(rgb_arr[i, j], rgb_palette)
                result_img[i, j] = color
                code_grid[i, j] = rgb_to_code.get(tuple(color), '??')
    return code_grid, result_img, alpha_arr, rgb_to_code

def diminish_colors(code_grid, result_img, rgb_to_code):
    h, w = code_grid.shape
    code_list = code_grid.flatten()
    codes_present = [code for code in code_list if code != '' and code is not None]
    if len(set(codes_present)) <= 1:
        return code_grid, result_img, False
    counts = Counter(codes_present)
    skip_codes = {"C01", "C02"}  # à adapter selon ta palette
    sorted_codes = [c for c, _ in sorted(counts.items(), key=lambda x: x[1]) if c not in skip_codes]
    if not sorted_codes:
        return code_grid, result_img, False
    min_code = sorted_codes[0]
    unique_codes = set(counts.keys()) - {min_code}
    unique_rgbs = [rgb for rgb, code in rgb_to_code.items() if code in unique_codes]
    if not unique_rgbs:
        return code_grid, result_img, False
    rgb_min = [rgb for rgb, code in rgb_to_code.items() if code == min_code][0]
    rgb_min_np = np.array(rgb_min)
    unique_rgbs_np = np.array(unique_rgbs)
    diffs = unique_rgbs_np - rgb_min_np
    dists = np.sum(diffs**2, axis=1)
    idx_closest = np.argmin(dists)
    rgb_closest = tuple(unique_rgbs_np[idx_closest])
    code_closest = rgb_to_code[rgb_closest]
    new_code_grid = code_grid.copy()
    new_result_img = result_img.copy()
    for i in range(h):
        for j in range(w):
            if code_grid[i, j] == min_code:
                new_code_grid[i, j] = code_closest
                new_result_img[i, j] = rgb_closest
    return new_code_grid, new_result_img, True
