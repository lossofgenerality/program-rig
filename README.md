# WASD Humanoid

A skinned 3D humanoid on a Mixamo-style armature, saved as glTF 2.0 for Blender and the web, with a playable page where you drive it around a small yard.

- **Play:** `index.html` (W A S D to move, Shift to run, Space to jump, drag to turn the camera, scroll to zoom; touch pad on phones)
- **Look:** `viewer.html` (turntable with every animation)

## What is in the repository

| Path | What it is |
|---|---|
| `index.html` | The playable WASD scene (three.js r128 from cdnjs, GLTFLoader from jsDelivr) |
| `viewer.html` | Turntable viewer for the model and its animations |
| `models/humanoid_rigged.glb` | The character: one skinned mesh, 22 bones, Idle / Walk / Run / Jump |
| `tools/make_rigged_humanoid.py` | The Python script that generates the model from scratch |
| `tools/requirements.txt` | Python dependency for the script (NumPy) |
| `LICENSE.md` | Licence notice |

## The model

- About 1.76 m tall, faces +Z, Y up, feet at y = 0, T-pose bind.
- One mesh of about 10,000 vertices with vertex colours (skin, shirt, trousers, belt, shoes, hair).
- 22 bones with Mixamo-style names, each pointing at its child: `Hips`, `Spine`, `Spine1`, `Spine2`, `Neck`, `Head`, `LeftShoulder`, `LeftArm`, `LeftForeArm`, `LeftHand`, `LeftUpLeg`, `LeftLeg`, `LeftFoot`, `LeftToeBase`, and the `Right…` side.
- Animations: `Idle` (3 s loop), `Walk` (1 s loop), `Run` (0.66 s loop), `Jump` (0.95 s, one-shot).
- Passes the Khronos glTF Validator with no errors or warnings.

### In Blender

File › Import › glTF 2.0 and pick `models/humanoid_rigged.glb`. It arrives as an Armature with the Body mesh bound to it; each animation is its own action (Action Editor or NLA).

## Host it on GitHub Pages

1. Create a public repository, for example `wasd-humanoid`.
2. Upload everything in this folder, keeping the `models` and `tools` folders (Add file › Upload files, then Commit).
3. Settings › Pages › Build and deployment: Source "Deploy from a branch", branch `main`, folder `/ (root)`, Save.
4. After a minute or two: `https://<your-username>.github.io/wasd-humanoid/`

Embed it elsewhere with an iframe, for example:

```html
<iframe src="https://<your-username>.github.io/wasd-humanoid/" title="WASD Humanoid"
  width="100%" height="900" style="width:100%; height:900px; border:0;" loading="lazy" allowfullscreen></iframe>
```

## Run it locally

The pages load the model with `fetch`, which browsers block for files opened straight from disk. Serve the folder instead:

```bash
python3 -m http.server 8000
```

then open http://localhost:8000 (game) or http://localhost:8000/viewer.html (viewer).

## Regenerate or change the model

```bash
pip install -r tools/requirements.txt
python3 tools/make_rigged_humanoid.py models/humanoid_rigged.glb
```

Proportions, colours, bone positions, skin-weight rules and the animation formulas are all plain numbers in the script.

## Licence

Design, code, model and text © 2026 Computational Classnotes · CC BY-NC-SA 4.0. See `LICENSE.md`.
