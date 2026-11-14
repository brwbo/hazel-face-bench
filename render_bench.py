import av, cv2, numpy as np
from pathlib import Path
from deepface import DeepFace
import sys
R=Path(sys.argv[1]);V=Path(sys.argv[2])
det=cv2.FaceDetectorYN.create(str(R/"face_detection_yunet_2026may.onnx"),"",(320,320),0.9,0.3,5000)
sf=cv2.FaceRecognizerSF.create(str(R/"face_recognition_sface_2021dec.onnx"),"")
M=["SFace","ArcFace","Facenet512"]
n=lambda v:(lambda a:a/(np.linalg.norm(a) or 1))(np.asarray(v,dtype=np.float32).ravel())
def emb(img,f):
    a=sf.alignCrop(img,f);o={"SFace":n(sf.feature(a))}
    for k in M[1:]:o[k]=n(DeepFace.represent(img_path=a,model_name=k,detector_backend="skip",align=False,enforce_detection=False)[0]["embedding"])
    return o
def clip(cid,t0,t1,step=0.25):
    c=av.open(str(next((V/cid).glob("*.webm"))));s=c.streams.video[0];nxt=t0
    for fr in c.decode(s):
        t=float(fr.pts*s.time_base)
        if t<t0 or t<nxt:continue
        if t>t1:break
        nxt=t+step;yield t,fr.to_ndarray(format="bgr24")
def first(cid):
    for t,img in clip(cid,0,10):
        h,w=img.shape[:2];det.setInputSize((w,h));_,f=det.detect(img)
        if f is not None and len(f):return emb(img,max(f,key=lambda r:r[2]*r[3]))
refs={c:first(c) for c in ["v01_head","v09_fi"]}
vw=cv2.VideoWriter("bench_faces.mp4",cv2.VideoWriter_fourcc(*"mp4v"),4,(854,480+150))
for cid,other in [("v01_head","v09_fi"),("v09_fi","v01_head")]:
    for t,img in clip(cid,6,26):
        img=cv2.resize(img,(854,480));h,w=img.shape[:2];det.setInputSize((w,h));_,f=det.detect(img)
        pan=np.full((150,854,3),(227,240,246),np.uint8)
        cv2.putText(pan,f"{cid}  t={t:4.1f}s   same person (ref=first frame)  |  different person ({other})",(12,24),cv2.FONT_HERSHEY_SIMPLEX,0.5,(40,40,35),1,cv2.LINE_AA)
        if f is not None and len(f):
            b=max(f,key=lambda r:r[2]*r[3]);x,y,bw,bh=[int(v) for v in b[:4]];cv2.rectangle(img,(x,y),(x+bw,y+bh),(80,120,79),2)
            e=emb(img,b)
            for i,k in enumerate(M):
                same=float(e[k]@refs[cid][k]);diff=float(e[k]@refs[other][k])
                y0=55+i*32;cv2.putText(pan,f"{k:<11} {same:5.2f}",(12,y0),cv2.FONT_HERSHEY_SIMPLEX,0.6,(40,40,35),1,cv2.LINE_AA)
                cv2.rectangle(pan,(210,y0-14),(210+int(max(0,same)*250),y0+2),(80,120,79),-1)
                cv2.putText(pan,f"{diff:5.2f}",(480,y0),cv2.FONT_HERSHEY_SIMPLEX,0.6,(40,40,35),1,cv2.LINE_AA)
                cv2.rectangle(pan,(540,y0-14),(540+int(max(0,diff)*250),y0+2),(56,94,176),-1)
        vw.write(np.vstack([img,pan]))
vw.release();print("done")
