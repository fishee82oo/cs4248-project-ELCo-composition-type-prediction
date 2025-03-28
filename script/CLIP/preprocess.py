import json
import requests
import os
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import regex
from PIL import Image, ImageDraw, ImageFont

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

if __name__ == '__main__':
    elco_df = pd.read_csv('data/ELCo.csv')
    emoji_desc_pair = get_emoji_desc_pair(elco_df)
    generate_elco_images(emoji_desc_pair, 'script/CLIP/NotoColorEmoji.ttf', save_dir='script/CLIP/images')
    