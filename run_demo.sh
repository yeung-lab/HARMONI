set -x
python main.py --video data/demo/giphy.gif --out_folder ./results/giphy --fps 3 --render --use_cached_dataset --ground_constraint --save_mesh --keep contains_only_both --ground_anchor child_bottom --tracker_type phalp --track_overwrite "{2: 'infant', 11: 'infant'}" --save_video --save_gif $1

