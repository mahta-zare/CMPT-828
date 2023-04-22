import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold_id', type=int, required=True)
    parser.add_argument('--mosaic_width', type=int, default=1300)
    parser.add_argument('--artifact_dir', default='/wdata')
    parser.add_argument('--train_dir', default='/data/train')
    return parser.parse_args()


def imsave(path, im_array, use_pillow=True, check_contrast=True):
    if use_pillow:
        im = Image.fromarray(im_array)
        im.save(path)
    else:
        # skimage's imsave is too slow..
        io.imsave(path, im_array, check_contrast=check_contrast)


def tile_fn_to_loc(tile_fn):
    tile_fn = os.path.splitext(tile_fn)[0]
    splits = tile_fn.split('_')
    prefix = f'{splits[0]}_{splits[1]}'
    x = int(splits[2])
    y = int(splits[3])
    return prefix, x, y


def tile_loc_to_fn(prefix, x, y):
    return f'{prefix}_{x}_{y}.tif'


pre_image_blacklist = [
    # Foundation error:
    # - Louisiana-East_Training_Public
    '10300100AF395C00_2_18_35.tif',  # building FN
    '10300100AF395C00_2_19_35.tif',  # building FN
    '10400100684A4B00_1_22_70.tif',  # building FN
    '10400100684A4B00_1_23_70.tif',  # building FN
    '10400100684A4B00_1_24_70.tif',  # building FN
    '10400100684A4B00_1_25_70.tif',  # building FN
    '10400100684A4B00_1_26_70.tif',  # building FN
    '10400100684A4B00_1_2_84.tif',  # building FN
    # Flood error:
    # - Germany_Training_Public
    '10500500C4DD7000_0_26_62.tif',  # warping
    '10500500C4DD7000_0_27_62.tif',  # warping
    '10500500C4DD7000_0_27_63.tif',  # flood road FP
    '10500500C4DD7000_0_27_64.tif',  # flood road FP
    '10500500C4DD7000_0_29_70.tif',  # warping
    '10500500C4DD7000_0_30_70.tif',  # warping
    # - Louisiana-East_Training_Public
    '10300100AF395C00_2_13_45.tif',  # flood road & building FN
    '10300100AF395C00_2_13_46.tif',  # flood building FN
    '10300100AF395C00_2_13_47.tif',  # flood road & building FN
    '10300100AF395C00_2_14_46.tif',  # flood building FN
    '10300100AF395C00_2_22_43.tif',  # flood road & building FN
    '105001001A0FFC00_0_12_13.tif',  # flood road FN
    '105001001A0FFC00_0_16_14.tif',  # flood road FN
    '105001001A0FFC00_0_17_15.tif',  # flood road FN
    '105001001A0FFC00_0_20_17.tif',  # flood road & building FN
    '10400100684A4B00_1_15_88.tif',  # flood road FN
    '10400100684A4B00_1_15_93.tif',  # flood road FN
    '10400100684A4B00_1_16_73.tif',  # flood road FN
    '10400100684A4B00_1_20_82.tif',  # flood building FN
    '10400100684A4B00_1_21_79.tif',  # flood building FN
    '10400100684A4B00_1_21_86.tif',  # flood building FN
    '10400100684A4B00_1_22_79.tif',  # flood building FN
    '10400100684A4B00_1_23_78.tif',  # flood road & building FN
    '10400100684A4B00_1_23_79.tif',  # flood road & building FN
]


def process(args, df, tile_prefix):
    out_root = os.path.join(args.artifact_dir, f'mosaics/train_{args.fold_id}')
    out_pre_dir = os.path.join(out_root, 'pre')

    out_road_dir = os.path.join(out_root, 'road')
    os.makedirs(out_pre_dir, exist_ok=True)

    os.makedirs(out_road_dir, exist_ok=True)
    out_rows = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        aoi = row['aoi']
        pre = row['pre-event image']
        if pre in pre_image_blacklist:
            continue
        pre_prefix, x, y = tile_fn_to_loc(pre)
        assert pre_prefix == tile_prefix

        # get path of Right tiles
        pre_right = tile_loc_to_fn(pre_prefix, x + 1, y)
        if pre_right in pre_image_blacklist:
            continue
        row_right = df[df['pre-event image'] == pre_right]
        if len(row_right) == 0:
            continue
        assert len(row_right) == 1
        pre_right_path = os.path.join(args.train_dir, aoi, 'PRE-event', pre_right)
        
        # get path of Bottom tiles
        pre_bottom = tile_loc_to_fn(pre_prefix, x, y + 1)
        if pre_bottom in pre_image_blacklist:
            continue
        row_bottom = df[df['pre-event image'] == pre_bottom]
        if len(row_bottom) == 0:
            continue
        assert len(row_bottom) == 1
        pre_bottom_path = os.path.join(args.train_dir, aoi, 'PRE-event', pre_bottom)


        # get path of Right-Bottom tiles
        pre_rb = tile_loc_to_fn(pre_prefix, x + 1, y + 1)
        if pre_rb in pre_image_blacklist:
            continue
        row_rb = df[df['pre-event image'] == pre_rb]
        if len(row_rb) == 0:
            continue
        assert len(row_rb) == 1
        pre_rb_path = os.path.join(args.train_dir, aoi, 'PRE-event', pre_rb)

        # load tiles
        pre_path = os.path.join(args.train_dir, aoi, 'PRE-event', pre)

        pre_image = io.imread(pre_path)
        
        pre_right_image = io.imread(pre_right_path)
        pre_bottom_image = io.imread(pre_bottom_path)
        pre_rb_image = io.imread(pre_rb_path)


        # mosaic!
        h, w, c = pre_image.shape
        zeros = np.zeros(shape=[h * 2, w * 2, c], dtype=np.uint8)

        crop_top = h - args.mosaic_width // 2
        crop_bottom = crop_top + args.mosaic_width
        crop_left = w - args.mosaic_width // 2
        crop_right = crop_left + args.mosaic_width
        

        pre_mosaic = zeros.copy()
        pre_mosaic[:h, :w] = pre_image
        pre_mosaic[:h, w:] = pre_right_image
        pre_mosaic[h:, :w] = pre_bottom_image
        pre_mosaic[h:, w:] = pre_rb_image
        pre_mosaic = pre_mosaic[crop_top:crop_bottom, crop_left:crop_right]

        label = f'{os.path.splitext(pre)[0]}.png'
        label_right = f'{os.path.splitext(pre_right)[0]}.png'
        label_bottom = f'{os.path.splitext(pre_bottom)[0]}.png'
        label_rb = f'{os.path.splitext(pre_rb)[0]}.png'


        road_mosaic = zeros.copy()
        road = io.imread(os.path.join(args.artifact_dir, 'masks_road/', aoi, label))
        road_right = io.imread(os.path.join(args.artifact_dir, 'masks_road/', aoi, label_right))
        road_bottom = io.imread(os.path.join(args.artifact_dir, 'masks_road/', aoi, label_bottom))
        road_rb = io.imread(os.path.join(args.artifact_dir, 'masks_road/', aoi, label_rb))
        road_mosaic[:h, :w] = road
        road_mosaic[:h, w:] = road_right
        road_mosaic[h:, :w] = road_bottom
        road_mosaic[h:, w:] = road_rb
        road_mosaic = road_mosaic[crop_top:crop_bottom, crop_left:crop_right]

        # save
        pre_fn = f'{os.path.splitext(pre)[0]}-rb.png'
        out_pre_path = os.path.join(out_pre_dir, pre_fn)
        imsave(out_pre_path, pre_mosaic)

        out_road_path = os.path.join(out_road_dir, pre_fn)
        imsave(out_road_path, road_mosaic, check_contrast=False)

        out_rows.append([
            os.path.abspath(out_pre_path),
            os.path.abspath(out_road_path)
        ])

    return out_rows


def main():
    args = parse_args()
    print(f'mosaicing fold-{args.fold_id}')

    rows = []
    df = pd.read_csv(os.path.join(args.artifact_dir, f'folds_v3/train_{args.fold_id}.csv'))
    tiles = ['10500500C4DD7000_0', '105001001A0FFC00_0', '10400100684A4B00_1', '10300100AF395C00_2']

    for i, tile_prefix in enumerate(tiles):
        print(f'mosaicing {tile_prefix} ({i + 1}/{len(tiles)})')
        df_filtered = df[df['pre-event image'].str.startswith(tile_prefix)]

        ret = process(args, df_filtered, tile_prefix)
        rows.extend(ret)

    df_out = pd.DataFrame(rows, columns=['pre', 'road'])
    out_dir = os.path.join(args.artifact_dir, f'mosaics/train_{args.fold_id}')
    os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(os.path.join(out_dir, 'mosaics.csv'), index=False)


if __name__ == '__main__':
    main()
