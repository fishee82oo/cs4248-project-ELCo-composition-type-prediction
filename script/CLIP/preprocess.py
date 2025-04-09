import json
import requests
import os
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import regex
from PIL import Image, ImageDraw, ImageFont
import math

# run in main directory

def preprocess_text(text):
    text = text.replace('\'\'', '').lower()
    split_text = regex.findall(r'\':?(.*?):?\'', text)
    return split_text

def split_emojis(s):
    return regex.findall(r"\X", s)

def preprocess_emoji(emojis):
    emojis_list = split_emojis(emojis)
    filtered = [c for c in emojis_list if c.strip() != '']
    return filtered

def get_emoji_desc_pair(elco_df):
    emoji_dict = {}

    for i, row in elco_df.iterrows():
        descs = preprocess_text(row["Description"])
        emojis = preprocess_emoji(row["EM"])
        if len(descs) != len(emojis):
            print(f"Error at: {i}, {descs}, {emojis}")
            break
        else:
            for j in range(len(descs)):
                emoji_dict[descs[j]] = emojis[j]
                
    emoji_desc_pair = {pair[0]: pair[1] for pair in set(emoji_dict.items())}
    return emoji_desc_pair

def emoji_to_image(em, desc, font_path, image_size=128, save_dir='images'):
    os.makedirs(save_dir, exist_ok=True)
    image = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(font_path, 109)
    draw.text((0, 0), em, font=font, embedded_color=True)
    image.save(os.path.join(os.curdir, save_dir, f"{desc}.png"))

def generate_elco_images(emoji_desc_pair, font_path, save_dir):
    for desc, em in emoji_desc_pair.items():
        emoji_to_image(em, desc, font_path, save_dir=save_dir)
    print("Images generated.")

def emoji_row_to_image(emojis, row_idx, font_path, final_size=224, save_dir='images_row'):
    os.makedirs(save_dir, exist_ok=True)
    emojis_list = preprocess_emoji(emojis)
    num_emojis = len(emojis_list)

    if num_emojis == 0:
        print(f"Skipping row {row_idx} due to no emojis.")
        return

    image = Image.new("RGB", (final_size, final_size), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Special case: 2 emojis
    if num_emojis == 2:
        cell_size = final_size // 2
        font = ImageFont.truetype(font_path, size=int(cell_size * 0.85))

        for i in range(2):
            emoji = emojis_list[i]
            x = i * cell_size

            bbox = draw.textbbox((0, 0), emoji, font=font, embedded_color=True)
            em_height = bbox[3] - bbox[1]
            y = (final_size - em_height) // 2

            draw.text((x, y), emoji, font=font, embedded_color=True)

        image.save(os.path.join(save_dir, f"row_{row_idx}.png"))
        return

    # General case
    grid_size = math.ceil(math.sqrt(num_emojis))
    cell_size = final_size // grid_size
    font = ImageFont.truetype(font_path, size=int(cell_size * 0.85))

    for idx, em in enumerate(emojis_list):
        row = idx // grid_size
        col = idx % grid_size
        x = col * cell_size
        y = row * cell_size

        bbox = draw.textbbox((0, 0), em, font=font, embedded_color=True)
        em_width = bbox[2] - bbox[0]
        em_height = bbox[3] - bbox[1]
        offset_x = (cell_size - em_width) // 2
        offset_y = (cell_size - em_height) // 2

        draw.text((x + offset_x, y + offset_y), em, font=font, embedded_color=True)

    image.save(os.path.join(save_dir, f"row_{row_idx}.png"))

if __name__ == '__main__':
    elco_df = pd.read_csv('data/ELCo.csv')
    emoji_desc_pair = get_emoji_desc_pair(elco_df)
    generate_elco_images(emoji_desc_pair, 'script/CLIP/NotoColorEmoji.ttf', save_dir='script/CLIP/images')

    for i, row in elco_df.iterrows():
        emoji_row_to_image(row["EM"], i, 'script/CLIP/NotoColorEmoji.ttf', save_dir='script/CLIP/images_row')
    