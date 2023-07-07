# vid=seedlings_fps1
# python main.py --images data/demo/${vid} --out_folder ./results/${vid} --fps 1 --render --use_cached_dataset --save_mesh --save_video --keep contains_child --ground_constraint --ground_anchor child_bottom


vid=vid
python main.py --images data/demo/${vid} --out_folder ./results/${vid} --fps 1 --render --use_cached_dataset --ground_constraint --save_mesh --keep contains_both --ground_anchor adult_bottom
