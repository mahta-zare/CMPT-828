/data/train  -> includes all the aois training data
Loissian_West	Germany

make_folds_v3.py -> /wdata/folds_v3 -> includes 5 set of val and train .csv files

prepare_road_masks.py -> /wdata/masks_raod -> includes masks of of roads. it is a 3 channel image of flooded_mask, not_flooded_mask, junction_mask

warp_post_images.py -> /wdata/warped_posts_test ->
warp_post_images.py -> /wdata/warped_posts_train -> post images modified to have the same shape/dims as the pre-event images

mosaic.py -> /wdata/mosaics/train_{fold_id} -> includes a mosaics.csv file with cols 'pre' and
'road' 
mosaic.py -> /wdata/mosaics/train_{fold_id}/pre -> pre-event mosaics
mosaic.py -> /wdata/mosaics/train_{fold_id}/road -> road mask mosaics.