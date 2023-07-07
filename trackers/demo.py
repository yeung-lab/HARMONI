import omegaconf
import joblib
from phalp import PHALP
import sys, os

sys.path.append('./trackers/phalp')
sys.path.append('/pasteur/u/zzweng/projects/HARMONI')

cfg = omegaconf.OmegaConf.load(os.path.abspath("./trackers/phalp_config.yaml"))
phalp_tracker = PHALP(cfg)

dataset = joblib.load('./results/vid/dataset.pt')
phalp_tracker.track(dataset)