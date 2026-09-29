# 日本語説明：従来の合成方法・速度・移動条件を調査する監査用スクリプトです。現在の同軸Thread検証の必須処理ではありません。
"""Audit current Bessel synthesis; leaves test.ipynb unchanged."""
import contextlib, io, json, platform, time
from pathlib import Path
import numpy as np
from scipy.special import j0
from threadpoolctl import threadpool_info
root=Path(__file__).resolve().parent
nb=json.loads((root/'test.ipynb').read_text(encoding='utf-8'))
ns={}
for i in (1,2,3,6,8,9,10,11,19):
    print('Setup cell', i, flush=True)
    with contextlib.redirect_stdout(io.StringIO()): exec(''.join(nb['cells'][i]['source']),ns)
N=ns['N']; dx=ns['dx']; X=ns['X']; Y=ns['Y']; R=ns['RANK']
B=ns['bessel_centered']; F=ns['bessel_fft']; Uy=ns['factor_ky']; Ux=ns['factor_kx']; G=ns['core_complex']
kx=ns['kx']; ky=ns['ky']; shifts=ns['SHIFT_LIST']; weights=ns['WEIGHT_LIST']; kr=ns['KR_LIST']; aperture=ns['APERTURE_RADIUS']

def err(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(a))
def direct_j0():
    out=np.zeros((N,N),np.complex64)
    for q,(sx,sy),w in zip(kr,shifts,weights):
        r=np.sqrt((X-sx)**2+(Y-sy)**2)
        b=(j0(q*r)*np.exp(-(r/aperture)**8)).astype(np.complex64)
        out+=w*b
    return out

def cached_roll():
    out=np.zeros((N,N),np.complex64)
    for b,(sx,sy),w in zip(B,shifts,weights):
        out+=w*np.roll(b,(int(round(sy/dx)),int(round(sx/dx))),axis=(0,1))
    return out

def cached_fft(s=shifts,w=weights):
    out=np.zeros((N,N),np.complex64)
    for f,(sx,sy),weight in zip(F,s,w):
        out+=weight*f*np.exp(-1j*ky[:,None]*sy)*np.exp(-1j*kx[None,:]*sx)
    return np.fft.ifft2(out,norm='ortho')

def make_core(s=shifts,w=weights,rank=R):
    uy=Uy[:,:rank]; ux=Ux[:,:rank]
    total=np.zeros((rank,rank),np.complex64)
    for g,(sx,sy),weight in zip(G,s,w):
        rx=ux.T@(np.exp(-1j*kx[:,None]*sx)*ux)
        ry=uy.T@(np.exp(-1j*ky[:,None]*sy)*uy)
        total+=weight*(ry@g[:rank,:rank]@rx.T)
    return total

def proposed(s=shifts,w=weights,rank=R):
    return np.fft.ifft2(Uy[:,:rank]@make_core(s,w,rank)@Ux[:,:rank].T,norm='ortho')
# Exactly equivalent reconstruction in spatial bases; no NxN inverse FFT online.
Vy=np.fft.ifft(Uy,axis=0,norm='ortho')
Vx=np.fft.ifft(Ux,axis=0,norm='ortho')
def spatial_basis(): return Vy@make_core()@Vx.T

print('Reference and baseline checks',flush=True)
t=time.perf_counter(); reference=ns['benchmark_theory'](); original_once=time.perf_counter()-t
methods={'optimized_direct_j0':direct_j0,'cached_integer_roll':cached_roll,'cached_full_fft':cached_fft,'proposed_current':proposed,'proposed_spatial_basis':spatial_basis}
results={}; timings={name:[] for name in methods}
for name,fn in methods.items():
    result=fn(); results[name]={'relative_field_error':err(reference,result),'relative_intensity_error':err(np.abs(reference)**2,np.abs(result)**2)}
for repeat in range(5):
    order=list(methods); order=order if repeat%2==0 else order[::-1]
    for name in order:
        start=time.perf_counter(); result=methods[name](); elapsed=time.perf_counter()-start
        timings[name].append(elapsed); del result
    print('Timing round',repeat+1,flush=True)
for name in methods:
    results[name].update(median_s=float(np.median(timings[name])),min_s=min(timings[name]),max_s=max(timings[name]))
print(json.dumps(results,indent=2),flush=True)

print('Shift/rank stress tests against periodic full Fourier reference',flush=True)
stress=[]
for maxshift in (0.0,0.125,0.5,1.0,1.5):
    for seed in (0,1,2):
        rng=np.random.default_rng(seed)
        s=rng.uniform(-maxshift,maxshift,(len(kr),2))
        w=rng.normal(size=len(kr))+1j*rng.normal(size=len(kr))
        exact=cached_fft(s,w)
        for rank in (20,40,80):
            approx=proposed(s,w,rank)
            stress.append(dict(maxshift=maxshift,seed=seed,rank=rank,relative_field_error=err(exact,approx)))
    print('Shift range',maxshift,'done',flush=True)

core=make_core()
def timed(fn):
    values=[]
    for _ in range(5):
        t=time.perf_counter(); val=fn(); values.append(time.perf_counter()-t); del val
    return float(np.median(values))
spectrum=Uy@core@Ux.T
steps={'shift_operators_core_sum':timed(make_core),'reconstruct_fourier':timed(lambda:Uy@core@Ux.T),'inverse_fft':timed(lambda:np.fft.ifft2(spectrum,norm='ortho')),'reconstruct_spatial_basis':timed(lambda:Vy@core@Vx.T)}
out={'environment':{'python':platform.python_version(),'numpy':np.__version__,'threadpools':threadpool_info()},'conditions':{'N':N,'rank':R,'modes':len(kr),'repeats':5,'shifts':shifts.tolist()},'original_direct_single_s':original_once,'baselines':results,'steps_median_s':steps,'stress':stress,'spatial_basis_equivalence':err(proposed(),spatial_basis()),'storage_bytes':{'complex64_full_library':int(B.nbytes),'core_and_real_bases':int(G.nbytes+Uy.nbytes+Ux.nbytes)}}
(root/'lsh_audit_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('Audit saved. Steps:',steps,flush=True)
