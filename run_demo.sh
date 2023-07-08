# vid=seedlings_fps1
# python main.py --images data/demo/${vid} --out_folder ./results/${vid} --fps 1 --render --use_cached_dataset --save_mesh --save_video --keep contains_child --ground_constraint --ground_anchor child_bottom


# vid=vid
# python main.py --images data/demo/${vid} --out_folder ./results/${vid} --fps 1 --render --use_cached_dataset --ground_constraint --save_mesh --keep contains_both --ground_anchor adult_bottom


# this is from https://giphy.com/gifs/mom-cleaning-dumping-toys-5pK2Rs57ZCACAh8Fxs
set -x
python main.py --video data/demo/giphy.gif --out_folder ./results/giphy_ground90_phalp --fps 1 --render --use_cached_dataset --ground_constraint --save_mesh --keep contains_only_both --ground_anchor child_bottom --save_gif --tracker_type phalp --track_overwrite "{2: 'infant', 11: 'infant'}" $1

# first run ./run_demo.sh --dryrun
# then ./run_demo.sh

