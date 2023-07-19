# python main.py --video /pasteur/data/1kd/brazil/raw/101_0.mp4 --out_folder ./results/101_0 --ground_constraint --keep contains_both --ground_anchor child_bottom --save_mesh $1

# python main.py --images ./results/101_0/images --out_folder ./results/101_0 --ground_constraint --keep contains_both --ground_anchor child_bottom --save_mesh $1


# set -x
python main.py --video data/demo/giphy.gif --out_folder ./results/giphy --fps 3 --render --use_cached_dataset --ground_constraint --save_mesh --keep contains_only_both --ground_anchor child_bottom --tracker_type phalp --track_overwrite "{2: 'infant', 11: 'infant'}" --save_video --save_gif $1

