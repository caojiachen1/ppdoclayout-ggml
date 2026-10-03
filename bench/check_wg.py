import numpy as np, struct

D = r"C:\Users\caoji\AppData\Local\Temp\dumpwg"

def load(name):
    with open(D + "/shapes.txt") as f:
        for line in f:
            t = line.split()
            if t[0] == name:
                shape = [int(x) for x in t[2:]]
                a = np.fromfile(f"{D}/{name}.bin", dtype=np.float32)
                return a.reshape(shape[::-1])  # ggml ne -> row-major (last dim fastest)
    raise KeyError(name)

B = np.array([[ 2.118156909942627,  2.146311759948731, -1.673836588859558, -1.004659295082092],
              [ 0.668844103813171, -1.755281925201416,  0.263708651065826,  0.544651985168457],
              [-0.479055941104889,  0.242195174098015,  1.683301687240601,  0.669751882553101],
              [-1.848919749259949,  3.216520309448242,  4.306654453277588, -4.437918186187744]])
G = np.array([[ 2.146311759948731,  0.933696150779724,  0.406179785728455],
              [-1.755281925201416,  1.310971140861511, -0.979127824306488],
              [ 1.023769974708557,  1.312750458717346,  1.683301687240601],
              [ 0.915698945522308, -2.015886306762695,  4.437918186187744]])
A = np.array([[ 0.175659596920013,  0.076416060328484],
              [-0.299846142530441,  0.223946735262871],
              [ 0.243508785963058,  0.312244206666946],
              [ 0.018418358638883, -0.040547512471676]])

xp = load("wg_xp")   # (48, 202, 202) channels, H, W  (CHW)
dt = load("wg_dt")   # (16, 48, 10000)
yt = load("wg_yt")   # (16, 48, 10000)
r  = load("wg_r")    # (48, 200, 200)
IC, IH_p, IW_p = xp.shape
OC = yt.shape[1]
T = dt.shape[2]
TW, TH = 100, 100

# load weight + bias from gguf
import gguf
rr = gguf.GGUFReader(r"D:\Codebase\ppdoclayout-ggml\bench\model-f32.gguf")
w = {t.name: t for t in rr.tensors}
gw = np.array(w["st0.b0.l0.weight"].weights if hasattr(w["st0.b0.l0.weight"], "weights") else w["st0.b0.l0.weight"].data, dtype=np.float32).reshape([int(x) for x in w["st0.b0.l0.weight"].shape][::-1])  # (48,48,3,3) oc,ic,kh,kw
gb = np.frombuffer(w["st0.b0.l0.bias"].data, dtype=np.float32)

# 1) input transform check on a few tiles
mx = 0
for tile in [0, 123, 4567, 9999]:
    ty, tx = tile // TW, tile % TW
    d = xp[:, 2*ty:2*ty+4, 2*tx:2*tx+4]              # (IC,4,4)
    dt_ref = np.einsum('ua,av,icd->uvic', B, B, d)    # d~[u][v][ic]
    got = dt[:, :, tile].reshape(16, IC)              # dt[p][ic][tile]
    ref = dt_ref.reshape(16, IC)
    mx = max(mx, np.abs(got - ref).max())
print("input transform max diff:", mx)

# 2) madd check
wg = np.einsum('ua,vb,ackl->uvck', G, G, gw).reshape(16, IC, OC)
yt_ref = np.einsum('ptic,pco->pto', dt.reshape(16, IC, T), wg)   # (16, OC, T)
print("madd max diff:", np.abs(yt.reshape(16, OC, T) - yt_ref).max())

# 3) output transform check
t1 = np.einsum('ui,pv->piv', A, yt_ref.reshape(16, OC, T))       # (16,2,OC,T)?? careful
Y = np.einsum('uio,uvp,vjo->opij', A, yt_ref.reshape(16, OC, T).transpose(0,2,1), A)
print(Y.shape)
