import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

img_dir = 'fashion_dataset/images/'
csv_path = 'fashion_dataset/styles.csv'
output_dir = 'fashion_dataset'

df = pd.read_csv(csv_path, on_bad_lines='skip')
df = df.dropna(subset=['gender', 'masterCategory', 'subCategory', 'articleType'])
df = df[df['masterCategory'] == 'Apparel']

selected_types = ['Tshirts', 'Dresses', 'Shoes', 'Bags', 'Skirts', 'Tops', 'Jeans', 'Shorts', 'Jackets', 'Boots']
df = df[df['articleType'].isin(selected_types)]

label2id = {k: i for i, k in enumerate(selected_types)}
df['label_id'] = df['articleType'].map(label2id)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_id'])

for split in ['train', 'val']:
    os.makedirs(f'fashion_dataset/images/{split}', exist_ok=True)
    os.makedirs(f'fashion_dataset/labels/{split}', exist_ok=True)

def process_split(split_df, split_name):
    for _, row in split_df.iterrows():
        filename = str(row['id']) + '.jpg'
        src = os.path.join(img_dir, filename)
        dst = os.path.join(f'fashion_dataset/images/{split_name}', filename)

        if os.path.exists(src):
            shutil.copy(src, dst)
            yolo_label = f"{row['label_id']} 0.5 0.5 1.0 1.0\n"
            label_file = filename.replace('.jpg', '.txt')
            with open(f'fashion_dataset/labels/{split_name}/{label_file}', 'w') as f:
                f.write(yolo_label)

process_split(train_df, 'train')
process_split(val_df, 'val')
