# Troubleshooting & Notes

## Common Issues

| Problem | Fix |
|---|---|
| `detectron2` build fails: `No module named 'torch'` | Use `--no-build-isolation` so the build sees the installed torch |
| `detectron2` build fails: `_ARRAY_API not found` | Pin `numpy<2` before building (torch 2.1.2 needs numpy 1.x) |
| `detectron2` build fails: `No module named 'pkg_resources'` | Pin `setuptools<70` (newer setuptools removed `pkg_resources`) |
| pyrender black screen / segfault | Set `PYOPENGL_PLATFORM=egl`, install `libegl1-mesa` |
| `torchgeometry` import error | Already patched — uses `kornia` instead |
| `sklearn.utils.linear_assignment_` error | Already patched — uses `scipy.optimize.linear_sum_assignment` |
| `scipy.ndimage.filters` deprecation | Already patched — uses `scipy.ndimage` directly |
| `torch.load` FutureWarning spam | Cosmetic only — add `weights_only=False` or suppress warnings |
| TF 1.15 fails on Python 3.8+ | Use Python 3.7 for audio venv |
| `pip<24` for audio venv | Newer pip drops Python 3.7 support |
| `dlib` build fails | Ensure `cmake` + C++ compiler are installed |
| OpenPose / PHALP downloads files at runtime | Pre-download `_DATA/` folder (see README) |
| `smplx` can't find body models | Check paths in `constants.py` match your `data/` mount |
| CUDA out of memory | Reduce `--fps` or process shorter clips |
| Body type classifier misclassifies tracks | Use `--dryrun` to inspect, then `--track_overwrite "{2: 'infant'}"` to fix |

## Version Pinning

These version constraints are important if you are setting up a custom environment (non-Docker).

| Package | Constraint | Reason |
|---|---|---|
| `setuptools` | `<70` | Newer removes `pkg_resources`, which torch 2.1.2 needs |
| `numpy` | `<2` | torch 2.1.2 compiled against numpy 1.x ABI |
| `torch` (visual) | `2.1.2` | Tested; needs CUDA 12.x wheels (`cu121`) |
| `torch` (audio) | `1.4.0` | Required by audio pipeline's x-vector extractor |
| `tensorflow` | `1.15.5` | Required by ALICE voice-type classifier |
| `python` (audio) | `3.7` | TF 1.15 does not support Python 3.8+ |
| `pip` (audio) | `<24` | Newer pip drops Python 3.7 |
| `detectron2` | from source | Must be compiled against the installed torch version |

## Rebuilding the Docker Image

If you need to modify or rebuild the image from the provided `Dockerfile`:

```bash
docker build -t harmoni .
```

Key build notes:
- The image uses **two separate Python venvs**: visual (Python 3.10) and audio (Python 3.7)
- `data/` is excluded via `.dockerignore` — it must be mounted at runtime
- `PYOPENGL_PLATFORM=egl` is set for headless rendering
- detectron2 requires `--no-build-isolation` to see the installed torch during build

## GPU vs CPU

The code auto-detects GPU availability. On a CPU-only machine, everything runs but will be significantly slower. No code changes are needed — `torch.cuda.is_available()` handles device selection automatically.
