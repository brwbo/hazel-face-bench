import itertools, json, sys, time
from pathlib import Path
import av, cv2, numpy as np

ROOT = Path(sys.argv[1])
VID = Path(sys.argv[2])
CLIPS = ["v01_head", "v07_name", "v08_name2", "v09_fi", "v10_fi2", "v13_radio"]
STEP = 1.0
MAXF = 40

det = cv2.FaceDetectorYN.create(str(ROOT / "face_detection_yunet_2026may.onnx"), "", (320, 320), 0.9, 0.3, 5000)
sface = cv2.FaceRecognizerSF.create(str(ROOT / "face_recognition_sface_2021dec.onnx"), "")

from deepface import DeepFace
DF = ["ArcFace", "Facenet512"]


def frames(path):
    c = av.open(str(path))
    s = c.streams.video[0]
    nxt = 0.0
    n = 0
    for fr in c.decode(s):
        t = float(fr.pts * s.time_base)
        if t < nxt:
            continue
        nxt = t + STEP
        yield t, fr.to_ndarray(format="bgr24")
        n += 1
        if n >= MAXF:
            break


def norm(v):
    v = np.asarray(v, dtype=np.float32).ravel()
    return v / (np.linalg.norm(v) or 1)


samples = []
timing = {"SFace": []}
for k in DF:
    timing[k] = []
for cid in CLIPS:
    f = next((VID / cid).glob("*.webm"))
    got = 0
    for t, img in frames(f):
        h, w = img.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(img)
        if faces is None or not len(faces):
            continue
        face = max(faces, key=lambda r: r[2] * r[3])
        if face[2] < 60:
            continue
        aligned = sface.alignCrop(img, face)
        embs = {}
        t0 = time.perf_counter()
        embs["SFace"] = norm(sface.feature(aligned))
        timing["SFace"].append(time.perf_counter() - t0)
        for k in DF:
            t0 = time.perf_counter()
            r = DeepFace.represent(img_path=aligned, model_name=k, detector_backend="skip", align=False, enforce_detection=False)
            embs[k] = norm(r[0]["embedding"])
            timing[k].append(time.perf_counter() - t0)
        samples.append((cid, t, embs))
        got += 1
    print(cid, "faces", got, flush=True)

models = ["SFace"] + DF
out = {}
for m in models:
    gen, imp = [], []
    for (ca, ta, ea), (cb, tb, eb) in itertools.combinations(samples, 2):
        s = float(ea[m] @ eb[m])
        if ca == cb:
            if abs(ta - tb) >= 3:
                gen.append(s)
        else:
            imp.append(s)
    gen, imp = np.array(gen), np.array(imp)
    ths = np.linspace(-0.2, 1, 601)
    far = np.array([(imp >= t).mean() for t in ths])
    frr = np.array([(gen < t).mean() for t in ths])
    i = int(np.argmin(np.abs(far - frr)))
    best = int(np.argmin(far + frr))
    dp = (gen.mean() - imp.mean()) / np.sqrt(0.5 * (gen.var() + imp.var()))
    out[m] = dict(
        genuine_mean=round(float(gen.mean()), 3), genuine_p5=round(float(np.percentile(gen, 5)), 3),
        impostor_mean=round(float(imp.mean()), 3), impostor_p95=round(float(np.percentile(imp, 95)), 3), impostor_max=round(float(imp.max()), 3),
        dprime=round(float(dp), 2), eer=round(float((far[i] + frr[i]) / 2), 4), eer_threshold=round(float(ths[i]), 3),
        best_threshold=round(float(ths[best]), 3), far_at_best=round(float(far[best]), 4), frr_at_best=round(float(frr[best]), 4),
        ms_per_face=round(1000 * float(np.median(timing[m])), 1), n_genuine=len(gen), n_impostor=len(imp),
    )
print(json.dumps(out, indent=1))
Path("results.json").write_text(json.dumps(out, indent=1))
