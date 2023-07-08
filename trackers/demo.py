import omegaconf
import joblib
from phalp import PHALP
import sys, os

sys.path.append('./trackers/phalp')
print(os.path.abspath("./"))
sys.path.append(os.path.abspath("./"))  # append path to HARMONI folder.
# sys.path.append('/pasteur/u/zzweng/projects/HARMONI')

dataset = joblib.load('./results/giphy_ground90/dataset.pt')

cfg = omegaconf.OmegaConf.load(os.path.abspath("./trackers/phalp_config.yaml"))
cfg.video.output_dir = 'results/giphy_ground90'
phalp_tracker = PHALP(cfg)
phalp_tracker.track(dataset)