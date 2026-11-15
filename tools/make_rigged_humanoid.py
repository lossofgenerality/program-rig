"""Build humanoid_rigged.glb: one smooth skinned body on a Mixamo-style armature (T-pose bind),
with Idle, Walk, Run and Jump. Faces +Z, Y up, ~1.76 m, feet at y=0. Colours are vertex colours."""
import json, math, struct, sys
import numpy as np

OUT = sys.argv[1]

# ---------------- skeleton (global T-pose positions) ----------------
BONES = [  # name, parent, global position
    ("Hips", None, (0, 0.95, 0)),
    ("Spine", "Hips", (0, 1.05, 0)),
    ("Spine1", "Spine", (0, 1.17, 0)),
    ("Spine2", "Spine1", (0, 1.29, 0)),
    ("Neck", "Spine2", (0, 1.47, 0)),
    ("Head", "Neck", (0, 1.56, 0)),
    ("LeftShoulder", "Spine2", (0.06, 1.42, 0)),
    ("LeftArm", "LeftShoulder", (0.19, 1.42, 0)),
    ("LeftForeArm", "LeftArm", (0.47, 1.42, 0)),
    ("LeftHand", "LeftForeArm", (0.73, 1.42, 0)),
    ("RightShoulder", "Spine2", (-0.06, 1.42, 0)),
    ("RightArm", "RightShoulder", (-0.19, 1.42, 0)),
    ("RightForeArm", "RightArm", (-0.47, 1.42, 0)),
    ("RightHand", "RightForeArm", (-0.73, 1.42, 0)),
    ("LeftUpLeg", "Hips", (0.095, 0.92, 0)),
    ("LeftLeg", "LeftUpLeg", (0.095, 0.50, 0)),
    ("LeftFoot", "LeftLeg", (0.095, 0.08, 0)),
    ("LeftToeBase", "LeftFoot", (0.095, 0.03, 0.12)),
    ("RightUpLeg", "Hips", (-0.095, 0.92, 0)),
    ("RightLeg", "RightUpLeg", (-0.095, 0.50, 0)),
    ("RightFoot", "RightLeg", (-0.095, 0.08, 0)),
    ("RightToeBase", "RightFoot", (-0.095, 0.03, 0.12)),
]
BI = {b[0]: i for i, b in enumerate(BONES)}
GPOS = {b[0]: np.array(b[2], float) for b in BONES}


def lin(h):
    h = h.lstrip('#')
    return [((int(h[k:k + 2], 16) / 255) ** 2.2) for k in (0, 2, 4)]


COL = dict(skin=lin("#D9A07C"), shirt=lin("#2E8A8F"), pants=lin("#2B3552"), shoe=lin("#1D1D24"),
           hair=lin("#2A1A12"), eye=lin("#15151C"), belt=lin("#3A2A1E"), lip=lin("#B56A5E"))

P, N, C, JI, JW, IDX = [], [], [], [], [], []


def blend(a, b, t):
    t = max(0.0, min(1.0, t)); t = t * t * (3 - 2 * t)
    return {a: 1 - t, b: t} if a != b else {a: 1.0}


def add_part(pos, faces, colfn, wfn, flat_normals_for=None):
    base = len(P)
    pos = np.asarray(pos, float)
    nrm = np.zeros_like(pos)
    for f in faces:
        a, b, c = pos[f[0]], pos[f[1]], pos[f[2]]
        n = np.cross(b - a, c - a)
        for k in f: nrm[k] += n
    ctr = pos.mean(axis=0)
    ln = np.linalg.norm(nrm, axis=1)
    bad = ln < 1e-9
    nrm[bad] = pos[bad] - ctr
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    for i, p in enumerate(pos):
        P.append(p); N.append(nrm[i]); C.append(colfn(p))
        w = sorted([kv for kv in wfn(p).items() if kv[1] > 1e-6], key=lambda kv: -kv[1])[:4]
        s = sum(v for _, v in w) or 1
        w = [(BI[k], v / s) for k, v in w] + [(0, 0.0)] * (4 - len(w))
        JI.append([k for k, _ in w]); JW.append([v for _, v in w])
    for f in faces: IDX.append([base + f[0], base + f[1], base + f[2]])


def loft(stations, seg=28):
    """stations: list of (center, u, v, a, b). Returns verts and faces with capped ends (caps use own verts)."""
    pos, faces = [], []
    for (c, u, v, a, b) in stations:
        for i in range(seg):
            t = 2 * math.pi * i / seg
            pos.append(c + u * a * math.cos(t) + v * b * math.sin(t))
    n = len(stations)
    for j in range(n - 1):
        for i in range(seg):
            a0 = j * seg + i; a1 = j * seg + (i + 1) % seg
            b0 = a0 + seg; b1 = a1 + seg
            faces += [[a0, b0, a1], [a1, b0, b1]]
    for end, flip in ((0, True), (n - 1, False)):
        c = stations[end][0]
        ci = len(pos); pos.append(c)
        ring = []
        for i in range(seg):
            ring.append(len(pos)); pos.append(pos[end * seg + i])
        for i in range(seg):
            f = [ci, ring[i], ring[(i + 1) % seg]]
            faces.append(f if not flip else [f[0], f[2], f[1]])
    return pos, faces


def ellipsoid(center, radii, rings=14, seg=22):
    pos, faces = [], []
    cx, cy, cz = center; rx, ry, rz = radii
    for j in range(rings + 1):
        th = math.pi * j / rings
        for i in range(seg + 1):
            ph = 2 * math.pi * i / seg
            pos.append(np.array([cx + rx * math.sin(th) * math.cos(ph), cy + ry * math.cos(th), cz + rz * math.sin(th) * math.sin(ph)]))
    for j in range(rings):
        for i in range(seg):
            a = j * (seg + 1) + i; b = a + seg + 1
            faces += [[a, a + 1, b], [a + 1, b + 1, b]]
    return pos, faces


def interp(keys, y):
    for k in range(len(keys) - 1):
        y0, a0 = keys[k][0], keys[k][1:]; y1, a1 = keys[k + 1][0], keys[k + 1][1:]
        if (y0 - y) * (y1 - y) <= 0:
            t = (y - y0) / (y1 - y0) if y1 != y0 else 0
            return [p + (q - p) * t for p, q in zip(a0, a1)]
    return list(keys[0][1:] if y < keys[0][0] else keys[-1][1:])


X, Y, Z = np.array([1., 0, 0]), np.array([0, 1., 0]), np.array([0, 0, 1.])

# ---- torso ----
TK = [(0.86, 0.15, 0.11), (0.95, 0.172, 0.122), (1.05, 0.158, 0.112), (1.12, 0.146, 0.105), (1.22, 0.165, 0.115),
      (1.32, 0.185, 0.12), (1.40, 0.196, 0.112), (1.45, 0.125, 0.088), (1.485, 0.058, 0.056)]
st = []
y = 0.86
while y <= 1.485 + 1e-6:
    a, b = interp(TK, y)
    st.append((np.array([0, y, 0.0]), X, Z, a, b)); y += 0.018


def torso_w(p):
    y = p[1]
    if y < 1.02: w = blend("Hips", "Spine", (y - 0.97) / 0.1)
    elif y < 1.13: w = blend("Spine", "Spine1", (y - 1.09) / 0.08)
    elif y < 1.27: w = blend("Spine1", "Spine2", (y - 1.21) / 0.08)
    else: w = blend("Spine2", "Neck", (y - 1.44) / 0.05)
    side = "LeftShoulder" if p[0] > 0 else "RightShoulder"
    k = max(0.0, min(1.0, (abs(p[0]) - 0.12) / 0.08)) * max(0.0, min(1.0, (y - 1.33) / 0.06))
    if k > 0:
        w = {b_: v * (1 - 0.6 * k) for b_, v in w.items()}; w[side] = w.get(side, 0) + 0.6 * k
    return w


def torso_c(p):
    y = p[1]
    if 0.965 < y < 0.99: return COL["belt"]
    return COL["shirt"] if y >= 0.99 else COL["pants"]


add_part(*loft(st), torso_c, torso_w)

# ---- neck and head ----
nst = [(np.array([0, y, 0.0]), X, Z, 0.05 - (y - 1.44) * 0.03, 0.05 - (y - 1.44) * 0.03) for y in np.arange(1.44, 1.605, 0.02)]
add_part(*loft(nst, 20), lambda p: COL["skin"], lambda p: blend("Neck", "Head", (p[1] - 1.55) / 0.04))
head_parts = [
    ((0, 1.665, 0.008), (0.105, 0.127, 0.115), "skin"),
    ((0, 1.70, -0.016), (0.112, 0.111, 0.119), "hair"),
    ((0.038, 1.675, 0.112), (0.014, 0.018, 0.01), "eye"),
    ((-0.038, 1.675, 0.112), (0.014, 0.018, 0.01), "eye"),
    ((0, 1.645, 0.125), (0.016, 0.022, 0.018), "skin"),
    ((0, 1.607, 0.11), (0.022, 0.006, 0.008), "lip"),
    ((0.104, 1.66, 0.0), (0.015, 0.03, 0.022), "skin"),
    ((-0.104, 1.66, 0.0), (0.015, 0.03, 0.022), "skin"),
]
for c_, r_, col in head_parts:
    add_part(*ellipsoid(c_, r_), lambda p, col=col: COL[col], lambda p: {"Head": 1.0})

# ---- arms (T-pose along ±X) ----
AK = [(0.15, 0.058), (0.19, 0.056), (0.33, 0.05), (0.47, 0.042), (0.60, 0.04), (0.73, 0.033), (0.745, 0.032)]
for sgn, pre in ((1, "Left"), (-1, "Right")):
    ast = []
    x = 0.15
    while x <= 0.745 + 1e-6:
        r = interp(AK, x)[0]
        ast.append((np.array([sgn * x, 1.42, 0.0]), Y, Z, r, r * 0.95)); x += 0.016

    def arm_w(p, pre=pre):
        x = abs(p[0])
        if x < 0.24: return blend(pre + "Shoulder", pre + "Arm", (x - 0.16) / 0.08)
        if x < 0.52: return blend(pre + "Arm", pre + "ForeArm", (x - 0.445) / 0.05)
        return blend(pre + "ForeArm", pre + "Hand", (x - 0.705) / 0.04)
    add_part(*loft(ast, 20), lambda p: COL["shirt"] if abs(p[0]) < 0.34 else COL["skin"], arm_w)
    add_part(*ellipsoid((sgn * 0.79, 1.42, 0.0), (0.062, 0.026, 0.044)), lambda p: COL["skin"], lambda p, pre=pre: {pre + "Hand": 1.0})
    add_part(*ellipsoid((sgn * 0.765, 1.418, 0.048), (0.028, 0.017, 0.016)), lambda p: COL["skin"], lambda p, pre=pre: {pre + "Hand": 1.0})

# ---- legs ----
LK = [(0.97, 0.088), (0.92, 0.088), (0.70, 0.075), (0.50, 0.062), (0.38, 0.062), (0.12, 0.046), (0.075, 0.044)]
for sgn, pre in ((1, "Left"), (-1, "Right")):
    lst = []
    y = 0.97
    while y >= 0.075 - 1e-6:
        r = interp(LK[::-1], y)[0]
        lst.append((np.array([sgn * 0.095, y, 0.0]), X, Z, r, r * 0.98)); y -= 0.018

    def leg_w(p, pre=pre):
        y = p[1]
        if y > 0.86: return blend(pre + "UpLeg", "Hips", (y - 0.89) / 0.08)
        if y > 0.3: return blend(pre + "Leg", pre + "UpLeg", (y - 0.475) / 0.05)
        return blend(pre + "Foot", pre + "Leg", (y - 0.09) / 0.05)
    add_part(*loft(lst, 22), lambda p: COL["pants"], leg_w)
    add_part(*ellipsoid((sgn * 0.095, 0.045, 0.045), (0.056, 0.045, 0.125)), lambda p: COL["shoe"],
             lambda p, pre=pre: blend(pre + "Foot", pre + "ToeBase", (p[2] - 0.06) / 0.06))

# ---------------- glTF assembly ----------------
bin_parts, buffer_views, accessors = [], [], []
offset = 0


def add_view(data, target=None):
    global offset
    pad = (-len(data)) % 4
    bin_parts.append(data + b"\x00" * pad)
    v = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
    if target: v["target"] = target
    buffer_views.append(v); offset += len(data) + pad
    return len(buffer_views) - 1


def acc(arr, kind, comp=5126, target=None, minmax=False):
    arr = np.ascontiguousarray(arr)
    a = {"bufferView": add_view(arr.tobytes(), target), "componentType": comp, "count": int(arr.shape[0]), "type": kind}
    if minmax:
        f = arr.reshape(arr.shape[0], -1)
        a["min"] = [float(x) for x in f.min(axis=0)]; a["max"] = [float(x) for x in f.max(axis=0)]
    accessors.append(a)
    return len(accessors) - 1


assert len(P) < 65535, len(P)
pos_a = acc(np.array(P, np.float32), "VEC3", target=34962, minmax=True)
nrm_a = acc(np.array(N, np.float32), "VEC3", target=34962)
col_a = acc(np.array(C, np.float32), "VEC3", target=34962)
jnt_a = acc(np.array(JI, np.uint8), "VEC4", comp=5121, target=34962)
wgt_a = acc(np.array(JW, np.float32), "VEC4", target=34962)
idx_a = acc(np.array(IDX, np.uint16).reshape(-1), "SCALAR", comp=5123, target=34963)

# ---- bone orientation: each bone's local +Y points at its child, as in Blender / Mixamo rigs ----
def qmul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


def qconj(q): return [-q[0], -q[1], -q[2], q[3]]


def qrot(q, v):
    r = qmul(qmul(q, [v[0], v[1], v[2], 0.0]), qconj(q)); return np.array(r[:3])


def qmat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def from_y(d):
    d = np.asarray(d, float); d = d / np.linalg.norm(d)
    c = d[1]
    if c > 1 - 1e-9: return [0.0, 0.0, 0.0, 1.0]
    if c < -1 + 1e-9: return [0.0, 0.0, 1.0, 0.0]  # 180 degrees about Z keeps +Z facing forward
    ax = np.cross([0, 1, 0], d); ax /= np.linalg.norm(ax)
    a = math.acos(c); s = math.sin(a / 2)
    return [ax[0] * s, ax[1] * s, ax[2] * s, math.cos(a / 2)]


AIM = {"Hips": "Spine", "Spine": "Spine1", "Spine1": "Spine2", "Spine2": "Neck", "Neck": "Head",
       "LeftShoulder": "LeftArm", "LeftArm": "LeftForeArm", "LeftForeArm": "LeftHand",
       "RightShoulder": "RightArm", "RightArm": "RightForeArm", "RightForeArm": "RightHand",
       "LeftUpLeg": "LeftLeg", "LeftLeg": "LeftFoot", "LeftFoot": "LeftToeBase",
       "RightUpLeg": "RightLeg", "RightLeg": "RightFoot", "RightFoot": "RightToeBase"}
PARENT = {b[0]: b[1] for b in BONES}
BREST = {}
for name, parent, g in BONES:
    if name in AIM:
        BREST[name] = from_y(GPOS[AIM[name]] - GPOS[name])
    else:  # end bones (Head, hands, toes) continue their parent's direction
        BREST[name] = BREST[parent] if name != "Head" else [0.0, 0.0, 0.0, 1.0]
IDQ = [0.0, 0.0, 0.0, 1.0]

nodes = []
bone_node = {}
for name, parent, g in BONES:
    Bp = BREST[parent] if parent else IDQ
    t = qrot(qconj(Bp), np.array(g) - (GPOS[parent] if parent else np.zeros(3)))
    r = qmul(qconj(Bp), BREST[name])
    nodes.append({"name": name, "translation": [float(v) for v in t], "rotation": [float(v) for v in r]})
    bone_node[name] = len(nodes) - 1
for name, parent, g in BONES:
    if parent:
        nodes[bone_node[parent]].setdefault("children", []).append(bone_node[name])
ibm = []
for name, _, g in BONES:
    Rt = qmat(BREST[name]).T
    tt = -Rt @ np.array(g, float)
    m = np.eye(4); m[:3, :3] = Rt; m[:3, 3] = tt
    ibm.append(m.T.reshape(-1))  # column-major
ibm_a = acc(np.array(ibm, np.float32), "MAT4")


def to_bone_space(name, q_world_aligned):
    """Animation rotations are authored as if every bone's rest frame matched the world axes;
    convert them to the oriented bones: q_local = conj(B_parent) * q * B_self."""
    Bp = BREST[PARENT[name]] if PARENT[name] else IDQ
    return qmul(qmul(qconj(Bp), q_world_aligned), BREST[name])
arm_root = len(nodes); nodes.append({"name": "Armature", "children": [bone_node["Hips"]]})
body = len(nodes); nodes.append({"name": "Body", "mesh": 0, "skin": 0})
root = None

# ---------------- animation ----------------
def qmul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


def qa(axis, ang):
    s = math.sin(ang / 2); return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(ang / 2)]


def qxyz(rx=0, ry=0, rz=0):  # rotate X, then Y, then Z (applied to the vector in that order)
    return qmul(qa((0, 0, 1), rz), qmul(qa((0, 1, 0), ry), qa((1, 0, 0), rx)))


HIPS_Y = 0.95
DOWN = 1.46  # arms hang about 6° out from the body


def bone_rots(p):
    g = lambda k: p.get(k, (0, 0, 0))
    out = {}
    hx, hy_, hz = g("Hips"); out["Hips"] = qxyz(hx, hy_, hz)
    sx, sy, sz = g("Spine"); out["Spine"] = qxyz(sx * 0.6, sy * 0.5, sz)
    out["Spine1"] = qxyz(sx * 0.4, sy * 0.5, 0)
    cx, cy, cz = g("Chest"); out["Spine2"] = qxyz(cx, cy, cz)
    out["Neck"] = qxyz(g("Head")[0] * 0.4, g("Head")[1] * 0.4, 0)
    out["Head"] = qxyz(g("Head")[0] * 0.6, g("Head")[1] * 0.6, g("Head")[2])
    for side, pre, sg in (("L", "Left", -1), ("R", "Right", 1)):
        rx, ry, rz = g("Shoulder" + side)
        out[pre + "Arm"] = qmul(qa((1, 0, 0), rx), qa((0, 0, 1), sg * DOWN + rz))
        e = g("Elbow" + side)[0]
        out[pre + "ForeArm"] = qa((0, 1, 0), e if side == "L" else -e)
        out[pre + "UpLeg"] = qxyz(*g("Hip" + side))
        out[pre + "Leg"] = qxyz(*g("Knee" + side))
        out[pre + "Foot"] = qxyz(*g("Ankle" + side))
    return out


animations = []
ANIM_BONES = ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm",
              "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg", "LeftFoot", "RightFoot"]


def make_clip(name, dur, pose_fn, fps=30):
    n = int(round(dur * fps))
    ts = np.array([i / fps for i in range(n + 1)], np.float32)
    poses = [pose_fn(float(t)) for t in ts]
    tin = acc(ts, "SCALAR", minmax=True)
    samplers = [{"input": tin, "output": acc(np.array([[0, p.get("hipsY", HIPS_Y), 0] for p in poses], np.float32), "VEC3"), "interpolation": "LINEAR"}]
    channels = [{"sampler": 0, "target": {"node": bone_node["Hips"], "path": "translation"}}]
    rots = [bone_rots(p) for p in poses]
    for b in ANIM_BONES:
        samplers.append({"input": tin, "output": acc(np.array([to_bone_space(b, r[b]) for r in rots], np.float32), "VEC4"), "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": bone_node[b], "path": "rotation"}})
    animations.append({"name": name, "samplers": samplers, "channels": channels})


def gait(t, T, A_leg, knee_max, A_arm, elbow, lean, bob, twist, knee_base=0.08):
    ph = 2 * math.pi * t / T
    p = {}
    for side, off in (("L", 0.0), ("R", math.pi)):
        f = ph + off
        thigh = -A_leg * math.sin(f) + 0.04
        knee = knee_base + knee_max * max(0.0, math.cos(f)) ** 2
        p["Hip" + side] = (thigh, 0, 0); p["Knee" + side] = (knee, 0, 0)
        p["Ankle" + side] = (-0.45 * (thigh + knee * 0.55) + 0.05, 0, 0)
        p["Shoulder" + side] = (A_arm * math.sin(f), 0, 0)
        p["Elbow" + side] = (elbow - 0.12 * max(0.0, -math.sin(f)), 0, 0)
    p["hipsY"] = HIPS_Y - bob * 0.5 + bob * 0.5 * math.cos(2 * ph)
    p["Hips"] = (0, twist * math.sin(ph), 0)
    p["Spine"] = (lean, -twist * 0.6 * math.sin(ph), 0)
    p["Chest"] = (lean * 0.4, -twist * 0.8 * math.sin(ph), 0)
    p["Head"] = (-lean * 0.9, twist * 0.5 * math.sin(ph), 0)
    return p


def idle(t, T=3.0):
    ph = 2 * math.pi * t / T
    return {"hipsY": HIPS_Y - 0.004 + 0.004 * math.cos(ph),
            "Spine": (0.015 * math.sin(ph), 0, 0), "Chest": (0.02 * math.sin(ph), 0, 0),
            "Head": (0.03 * math.sin(ph * 0.5), 0.12 * math.sin(ph * 0.5), 0),
            "ShoulderL": (0.04 * math.sin(ph), 0, 0.02 * math.sin(ph)), "ShoulderR": (-0.04 * math.sin(ph), 0, -0.02 * math.sin(ph)),
            "ElbowL": (-0.12, 0, 0), "ElbowR": (-0.12, 0, 0),
            "HipL": (0.02, 0, 0.02), "HipR": (0.02, 0, -0.02), "KneeL": (0.04, 0, 0), "KneeR": (0.04, 0, 0),
            "AnkleL": (-0.05, 0, 0), "AnkleR": (-0.05, 0, 0)}


JUMP_KEYS = [
    (0.00, dict(hipsY=HIPS_Y, ElbowL=(-0.12, 0, 0), ElbowR=(-0.12, 0, 0))),
    (0.20, dict(hipsY=HIPS_Y - 0.18, Spine=(0.35, 0, 0), Head=(-0.25, 0, 0), HipL=(-0.95, 0, 0), HipR=(-0.95, 0, 0), KneeL=(1.55, 0, 0), KneeR=(1.55, 0, 0),
                AnkleL=(-0.55, 0, 0), AnkleR=(-0.55, 0, 0), ShoulderL=(0.75, 0, 0), ShoulderR=(0.75, 0, 0), ElbowL=(-0.3, 0, 0), ElbowR=(-0.3, 0, 0))),
    (0.32, dict(hipsY=HIPS_Y, Spine=(-0.05, 0, 0), HipL=(0.05, 0, 0), HipR=(0.05, 0, 0), KneeL=(0.05, 0, 0), KneeR=(0.05, 0, 0),
                AnkleL=(0.45, 0, 0), AnkleR=(0.45, 0, 0), ShoulderL=(-2.6, 0, 0.2), ShoulderR=(-2.6, 0, -0.2), ElbowL=(-0.2, 0, 0), ElbowR=(-0.2, 0, 0))),
    (0.52, dict(hipsY=HIPS_Y, Spine=(0.15, 0, 0), HipL=(-0.75, 0, 0), HipR=(-0.55, 0, 0), KneeL=(1.1, 0, 0), KneeR=(0.9, 0, 0),
                AnkleL=(0.1, 0, 0), AnkleR=(0.1, 0, 0), ShoulderL=(-1.2, 0, 0.4), ShoulderR=(-1.2, 0, -0.4), ElbowL=(-0.6, 0, 0), ElbowR=(-0.6, 0, 0))),
    (0.72, dict(hipsY=HIPS_Y - 0.14, Spine=(0.3, 0, 0), Head=(-0.2, 0, 0), HipL=(-0.8, 0, 0), HipR=(-0.8, 0, 0), KneeL=(1.25, 0, 0), KneeR=(1.25, 0, 0),
                AnkleL=(-0.45, 0, 0), AnkleR=(-0.45, 0, 0), ShoulderL=(-0.5, 0, 0.3), ShoulderR=(-0.5, 0, -0.3), ElbowL=(-0.4, 0, 0), ElbowR=(-0.4, 0, 0))),
    (0.95, dict(hipsY=HIPS_Y, ElbowL=(-0.12, 0, 0), ElbowR=(-0.12, 0, 0))),
]


def jump(t):
    for k in range(len(JUMP_KEYS) - 1):
        t0, a = JUMP_KEYS[k]; t1, b = JUMP_KEYS[k + 1]
        if t0 <= t <= t1:
            u = (t - t0) / (t1 - t0); u = u * u * (3 - 2 * u)
            out = {}
            for key in set(a) | set(b):
                d = HIPS_Y if key == "hipsY" else (0, 0, 0)
                va, vb = a.get(key, d), b.get(key, d)
                out[key] = va + (vb - va) * u if key == "hipsY" else tuple(x + (y - x) * u for x, y in zip(va, vb))
            return out
    return dict(JUMP_KEYS[-1][1])


make_clip("Idle", 3.0, idle)
make_clip("Walk", 1.0, lambda t: gait(t, 1.0, A_leg=0.45, knee_max=0.75, A_arm=0.38, elbow=-0.28, lean=0.05, bob=0.035, twist=0.09))
make_clip("Run", 0.66, lambda t: gait(t, 0.66, A_leg=0.78, knee_max=1.45, A_arm=0.75, elbow=-1.3, lean=0.24, bob=0.06, twist=0.14, knee_base=0.25))
make_clip("Jump", 0.95, jump)

gltf = {
    "asset": {"version": "2.0", "generator": "Computational Classnotes · rigged humanoid builder",
              "copyright": "© 2026 Computational Classnotes · CC BY-NC-SA 4.0"},
    "scene": 0, "scenes": [{"name": "Scene", "nodes": [arm_root, body]}],
    "nodes": nodes,
    "meshes": [{"name": "Body", "primitives": [{"attributes": {"POSITION": pos_a, "NORMAL": nrm_a, "COLOR_0": col_a, "JOINTS_0": jnt_a, "WEIGHTS_0": wgt_a},
                                                 "indices": idx_a, "material": 0}]}],
    "materials": [{"name": "BodyVertexColour", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 0.62}}],
    "skins": [{"name": "Armature", "joints": [bone_node[b[0]] for b in BONES], "inverseBindMatrices": ibm_a, "skeleton": bone_node["Hips"]}],
    "accessors": accessors, "bufferViews": buffer_views, "buffers": [{"byteLength": offset}], "animations": animations,
}
js = json.dumps(gltf, separators=(",", ":")).encode(); js += b" " * ((-len(js)) % 4)
binary = b"".join(bin_parts)
with open(OUT, "wb") as fh:
    fh.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binary)))
    fh.write(struct.pack("<II", len(js), 0x4E4F534A)); fh.write(js)
    fh.write(struct.pack("<II", len(binary), 0x004E4942)); fh.write(binary)
print("wrote", OUT, len(P), "vertices", len(IDX), "triangles", len(BONES), "bones", [a["name"] for a in animations])
