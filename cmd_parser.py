import configargparse


def parse_config(argv=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'HARMONI configurations'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description)
    parser.add_argument('-c', '--config', is_config_file=True, default='data/cfgs/harmoni.yaml',
                        help='config file path')
    
    parser.add_argument('--images', type=str, default='', help='Path to the input images folder')
    parser.add_argument('--video', type=str, default='', help='Path to the input video')
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame of the input video')
    parser.add_argument('--end_frame', type=int, default=100, help='End frame of the input video')
    parser.add_argument('--pipeline', type=int, choices=[1, 2], default=1, 
                        help='Pipeline 1: run openpose, run tracker, classify bbox. Pipeline 2: grounded dino, run tracker.')
    
    parser.add_argument('--out_folder', type=str, default='', help='Where the results are saved')
    parser.add_argument('--fps', type=int, default=1, help='FPS of the input video')
    
    parser.add_argument('--use_cached_dataset', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--camera_focal', type=int, default=2000)
    parser.add_argument('--dryrun', default=False, action='store_true')
    parser.add_argument('--tracker_type', type=str, default='phalp', choices=['phalp', 'dummy'], 
                        help='If dummy, each person is a track.')
    parser.add_argument('--track_overwrite', type=str, default='{}', help='Overwrite the tracking results. e.g. {4: `adult`}')

    parser.add_argument('--hps', type=str, default='dapa', choices=['dapa', 'cliff'], 
                        help='The hps model being used')
    parser.add_argument('--smpl_model', type=str, default='smpla', choices=['smpl_smil', 'smpla'], 
                        help='If smpl_smil, use the smil for infant. Otherwise, use smpla for both adult and child.')
    parser.add_argument('--kid_age', type=float, default=1.0, help='The age offset of the kid. Only used when smpla is used.')
    
    parser.add_argument('--run_smplify', default=False, action='store_true')
    parser.add_argument('--ground_constraint', default=False, action='store_true')
    parser.add_argument('--ground_weight', type=float, default=500.0)
    parser.add_argument('--ground_anchor', default='adult_bottom', choices=['adult_bottom', 'child_bottom'])
    # parser.add_argument('--get_ground_normal_from', type=str, choices=['depth', 'user_input'], default='depth')
    parser.add_argument('--smplify_iters', type=int, default=10)

    parser.add_argument('--add_downstream', default=False, action='store_true')

    parser.add_argument('--render_only', default=False, action='store_true', 
                        help='Only render the results. Assuming there is already a results.pt file in the result folder.')
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--renderer', default="pyrender", choices=['pyrender', 'sim3drender'], help='Which renderer to use.')
    parser.add_argument('--top_view', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_gif', default=False, action='store_true')
    parser.add_argument('--save_mesh', default=False, action='store_true')
    parser.add_argument('--keep', default='all', 
                        choices=['all', 'contains_child', 'contains_adult', 'contains_both',
                                 'contains_only_both'], help='Which results to keep')
    
    if argv:
        args = parser.parse_args(args=argv.split(' '))
    else:
        args = parser.parse_args()

    return args
