"""Locked RETFound official-pretrain frozen encoder + MLP BRSET source arm."""
from __future__ import annotations
import argparse, copy, csv, hashlib, io, json, os, random, sys, time, zipfile
from pathlib import Path

PROJECT=Path(os.environ['RETINAL_PROJECT_ROOT']).resolve()
BASE=PROJECT/'revision_2026-11-06'
WEIGHT=Path(os.environ['RETFOUND_OFFICIAL_WEIGHT']).resolve()
EXPECTED='e1e4f66a1b792eeb6e2efaf158f33be35c8255f36b3d17ed67cd5129da246485'
ZIP=Path(os.environ['BRSET_LICENSED_ZIP']).resolve()
TARGET_ZIP=Path(os.environ['MBRSET_LICENSED_ZIP']).resolve()
SPLIT=PROJECT/'outputs/splits/BRSET_source_split_v2.csv'
TARGET_SPLIT=PROJECT/'outputs/audit/mBRSET_master_manifest_v2.csv'
OUT=BASE/'unified_experiment/r11_matched_source_20261112'
CACHE=BASE/'private_predictions/r11_official_frozen_features_20261112'
SEEDS=(20260906,20260907,20260908)

USE_ENV_TORCH=os.environ.get('RETINAL_USE_ENV_TORCH') == '1'
if USE_ENV_TORCH and ('--mode' not in sys.argv or sys.argv[sys.argv.index('--mode')+1:sys.argv.index('--mode')+2] != ['smoke']):
    raise ValueError('RETINAL_USE_ENV_TORCH is permitted only for --mode smoke in this candidate copy')
if not USE_ENV_TORCH:
    CUDA_VENDOR=BASE/'analyses/torch_cuda_vendor';DEPS_VENDOR=BASE/'analyses/gpu_deps_vendor'
    sys.path.insert(0,str(CUDA_VENDOR));sys.path.insert(0,str(DEPS_VENDOR))
    sys.path.insert(2,str(BASE/'analyses/torch_vendor'))
    if os.name=='nt':os.add_dll_directory(str(CUDA_VENDOR/'torch/lib'))
import torch
torch.serialization.add_safe_globals([argparse.Namespace])
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
sys.path.insert(0,str(BASE/'reproducibility/vendor/RETFound'))
import models_vit
from util.datasets import build_transform
from types import SimpleNamespace

TF_ARGS=SimpleNamespace(input_size=224,color_jitter=None,aa='rand-m9-mstd0.5-inc1',
                        reprob=0.25,remode='pixel',recount=1)
TRANSFORM=build_transform('val',TF_ARGS)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()

def frames():
    import pandas as pd
    d=pd.read_csv(SPLIT)
    assert len(d)==2574
    counts=d.groupby('split').patient_id.nunique().to_dict()
    assert counts=={'TRAIN':939,'VALIDATION':202,'FROZEN_SOURCE_TEST':202},counts
    groups={k:set(g.patient_id.astype(str)) for k,g in d.groupby('split')}
    assert all(not groups[a]&groups[b] for a in groups for b in groups if a!=b)
    assert d.groupby('patient_id').referable_dr_patient.nunique().max()==1
    assert d.groupby('split').size().to_dict()=={'TRAIN':1802,'VALIDATION':385,'FROZEN_SOURCE_TEST':387}
    return d

class BrsetImages(Dataset):
    def __init__(self,frame):
        self.frame=frame.reset_index(drop=True);self.z=zipfile.ZipFile(ZIP)
    def __len__(self):return len(self.frame)
    def __getitem__(self,i):
        r=self.frame.iloc[i]
        member='brazilian-ophthalmological_1.0.2/'+str(r.image_path).replace('\\','/')
        with Image.open(io.BytesIO(self.z.read(member))) as im:x=TRANSFORM(im.convert('RGB'))
        return x,int(r.referable_dr_image),str(r.patient_id),int(r.referable_dr_patient),str(r.image_id)

class MbrsetImages(Dataset):
    def __init__(self,frame):
        self.frame=frame.reset_index(drop=True);self.z=zipfile.ZipFile(TARGET_ZIP)
    def __len__(self):return len(self.frame)
    def __getitem__(self,i):
        r=self.frame.iloc[i]
        member='mbrset_1.0.0/'+str(r.image_path).replace('\\','/')
        with Image.open(io.BytesIO(self.z.read(member))) as im:x=TRANSFORM(im.convert('RGB'))
        return x,int(r.referable_dr_image),str(r.patient_id),int(r.referable_dr_patient),str(r.image_id)

def encoder():
    assert WEIGHT.stat().st_size==3952489221 and sha(WEIGHT)==EXPECTED
    cp=torch.load(WEIGHT,map_location='cpu',weights_only=True,mmap=True)
    assert 'model' in cp and len(cp['model'])==398
    model=models_vit.RETFound_mae(img_size=224,num_classes=2,drop_path_rate=0.2,global_pool=False)
    state=cp['model'].copy()
    for k in ('head.weight','head.bias'):
        if k in state and state[k].shape!=model.state_dict()[k].shape:del state[k]
    result=model.load_state_dict(state,strict=False)
    assert result.missing_keys==['head.weight','head.bias'],result.missing_keys
    assert all(k.startswith(('decoder','mask_token')) for k in result.unexpected_keys),result.unexpected_keys[:10]
    model.eval()
    for p in model.parameters():p.requires_grad=False
    return model,{'missing':result.missing_keys,'unexpected_count':len(result.unexpected_keys),
                  'unexpected_first':result.unexpected_keys[:10],
                  'encoder_loaded_keys':sum(k in state for k in model.state_dict() if not k.startswith('head.'))}

def gpu_smoke():
    model,loading=encoder()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    d=frames();ds=BrsetImages(d[d.split=='TRAIN'].iloc[:2]);x,*_=next(iter(DataLoader(ds,batch_size=2,num_workers=0)))
    model.to(device)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    t=time.perf_counter()
    with torch.inference_mode():feat=model.forward_features(x.to(device)).float().squeeze(1)
    assert tuple(feat.shape)==(2,1024) and torch.isfinite(feat).all()
    output=Path(os.environ['RETINAL_SMOKE_OUTPUT']).resolve() if os.environ.get('RETINAL_SMOKE_OUTPUT') else OUT/'official_weight_gpu_smoke.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    info={'status':'REAL_OFFICIAL_WEIGHT_FORWARD_PASS','weight_sha256':EXPECTED,
          'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,'torch':torch.__version__,'cuda':torch.version.cuda,
          'batch':2,'feature_shape':list(feat.shape),'peak_reserved_mib':round(torch.cuda.max_memory_reserved()/2**20,2) if device=='cuda' else None,
          'seconds':round(time.perf_counter()-t,3),'load':loading}
    output.write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info))

def extract():
    d=frames();model,loading=encoder();model.cuda();model.eval()
    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    for split in ('TRAIN','VALIDATION','FROZEN_SOURCE_TEST'):
        target=CACHE/(split.lower()+'.npz')
        if target.exists():
            print('EXISTS',split,target);continue
        ds=BrsetImages(d[d.split==split]);loader=DataLoader(ds,batch_size=2,shuffle=False,num_workers=0,pin_memory=True)
        feats=[];yi=[];yp=[];pid=[];iid=[];t=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for n,(x,yimg,p,ypt,im) in enumerate(loader):
                f=model.forward_features(x.cuda(non_blocking=True)).float().squeeze(1).cpu().numpy()
                feats.append(f);yi.extend(yimg.tolist());yp.extend(ypt.tolist());pid.extend(p);iid.extend(im)
                if (n+1)%100==0:print('EXTRACT',split,n+1,'/',len(loader),flush=True)
        a=np.concatenate(feats).astype('float32')
        assert a.shape==(len(ds),1024) and np.isfinite(a).all()
        np.savez_compressed(target,features=a,y_image=np.array(yi,dtype='int8'),
                            y_patient=np.array(yp,dtype='int8'),patient_id=np.array(pid),image_id=np.array(iid))
        meta={'split':split,'images':len(ds),'patients':len(set(pid)),'feature_shape':list(a.shape),
              'positive_patients':len({p for p,y in zip(pid,yp) if y}),
              'weight_sha256':EXPECTED,'preprocessing':'official RETFound val bicubic resize256 centercrop224 ImageNet norm',
              'feature':'pretrained CLS token after pretrained norm, global_pool=False',
              'precision':'float32 forward','peak_reserved_mib':round(torch.cuda.max_memory_reserved()/2**20,2),
              'seconds':round(time.perf_counter()-t,2),'cache_sha256':sha(target),'load':loading}
        (OUT/(split.lower()+'_feature_manifest.json')).write_text(json.dumps(meta,indent=2),encoding='utf-8')
        print('DONE',split,json.dumps({k:meta[k] for k in ('images','patients','seconds','peak_reserved_mib')}),flush=True)

class Head(nn.Module):
    def __init__(self):
        super().__init__();self.bn=nn.BatchNorm1d(1024,eps=1e-6,momentum=.99)
        self.fc=nn.Sequential(nn.Linear(1024,512),nn.GELU(),nn.Dropout(.1),
                              nn.Linear(512,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,1))
        for layer in self.fc:
            if isinstance(layer,nn.Linear):
                nn.init.constant_(layer.bias,0.0)
                nn.init.normal_(layer.weight,mean=0.0,std=0.01)
    def forward(self,x):return self.fc(self.bn(x)).squeeze(-1)

def readcache(split):
    p=CACHE/(split.lower()+'.npz')
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def patient_score(logits,c):
    p=torch.sigmoid(torch.as_tensor(logits,dtype=torch.float64)).numpy()
    out={}
    for pid,y,prob in zip(c['patient_id'],c['y_patient'],p):
        if pid in out:assert out[pid][0]==int(y);out[pid]=(int(y),max(out[pid][1],float(prob)))
        else:out[pid]=(int(y),float(prob))
    return np.array([x[0] for x in out.values()]),np.array([x[1] for x in out.values()])

def patient_rows(logits,c,seed,cohort):
    p=torch.sigmoid(torch.as_tensor(logits,dtype=torch.float64)).numpy()
    out={}
    for pid,y,prob in zip(c['patient_id'],c['y_patient'],p):
        key=str(pid)
        if key in out:
            assert out[key]['label']==int(y)
            out[key]['probability']=max(out[key]['probability'],float(prob))
            out[key]['images']+=1
        else:out[key]={'cohort':cohort,'seed':seed,'patient_id':key,'label':int(y),
                       'probability':float(prob),'images':1}
    return list(out.values())

def predict(head,c):
    head.eval()
    with torch.inference_mode():return head(torch.from_numpy(c['features'])).float().numpy()

def train():
    train=readcache('TRAIN');val=readcache('VALIDATION')
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    assert len(train['features'])==1802 and len(val['features'])==385
    for seed in SEEDS:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        h=Head();assert sum(p.numel() for p in h.parameters())==658433
        opt=torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.05)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100)
        loader=DataLoader(TensorDataset(torch.from_numpy(train['features']),torch.from_numpy(train['y_image'].astype('float32'))),
                          batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed))
        best=-1;best_epoch=None;stale=0;history=[]
        for epoch in range(1,101):
            h.train();losses=[]
            for x,y in loader:
                opt.zero_grad();logits=h(x);loss=nn.functional.binary_cross_entropy_with_logits(logits,y)
                loss.backward();opt.step();losses.append(float(loss.detach()))
            sched.step();y,p=patient_score(predict(h,val),val);auc=float(roc_auc_score(y,p))
            history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'val_patient_auroc':auc,'lr':opt.param_groups[0]['lr']})
            print('TRAIN',seed,epoch,round(auc,6),flush=True)
            if auc>best+1e-10:
                best=auc;best_epoch=epoch;stale=0
                torch.save({'head':copy.deepcopy(h.state_dict()),'seed':seed,'best_epoch':epoch,'val_patient_auroc':auc},OUT/f'head_seed{seed}_best.pth')
            else:stale+=1
            if stale>=15:break
        with (OUT/f'head_seed{seed}_history.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
        cp=OUT/f'head_seed{seed}_best.pth'
        info={'seed':seed,'best_epoch':best_epoch,'val_patient_auroc':best,'epochs_ran':len(history),
              'head_sha256':sha(cp),'trainable_parameters':658433,'weight_sha256':EXPECTED,
              'optimizer':'AdamW lr=1e-3 weight_decay=.05 batch256 cosine100 patience15',
              'selection':'BRSET validation patient max-image probability AUROC only'}
        (OUT/f'head_seed{seed}_selection.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
        print('SELECTED',json.dumps(info),flush=True)

def test():
    c=readcache('FROZEN_SOURCE_TEST');assert len(c['features'])==387
    rows=[];predrows=[]
    for seed in SEEDS:
        path=OUT/f'head_seed{seed}_best.pth';sel=OUT/f'head_seed{seed}_selection.json'
        assert path.is_file() and sel.is_file() and sha(path)==json.loads(sel.read_text())['head_sha256']
        cp=torch.load(path,map_location='cpu',weights_only=True);h=Head();h.load_state_dict(cp['head'],strict=True)
        logits=predict(h,c)
        predrows.extend(patient_rows(logits,c,seed,'BRSET_SOURCE_TEST'))
        y,p=patient_score(logits,c);auc=float(roc_auc_score(y,p))
        rng=np.random.default_rng(seed+1000);pos=np.flatnonzero(y);neg=np.flatnonzero(1-y);bs=[]
        for _ in range(2000):
            ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
            bs.append(roc_auc_score(y[ix],p[ix]))
        lo,hi=np.quantile(bs,[.025,.975])
        rows.append({'seed':seed,'source_test_patients':len(y),'positive':int(y.sum()),'auroc':auc,
                     'ci_low':float(lo),'ci_high':float(hi),'bootstrap':2000,'head_sha256':sha(path)})
    with (OUT/'brset_source_test_results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    CACHE.mkdir(parents=True,exist_ok=True)
    with (CACHE/'brset_source_test_patient_predictions_PRIVATE.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(predrows[0]));w.writeheader();w.writerows(predrows)
    print('SOURCE_TEST',json.dumps(rows),flush=True)

def external():
    import pandas as pd
    assert all((OUT/f'head_seed{s}_selection.json').exists() for s in SEEDS)
    d=pd.read_csv(TARGET_SPLIT)
    d=d[(d.split=='FROZEN_TARGET_TEST') & (d.exclusion_status=='INCLUDED')].copy()
    assert d.patient_id.nunique()==400 and len(d)==1498,(d.patient_id.nunique(),len(d))
    assert d.groupby('patient_id').referable_dr_patient.nunique().max()==1
    model,loading=encoder();model.cuda();model.eval()
    ds=MbrsetImages(d);loader=DataLoader(ds,batch_size=2,shuffle=False,num_workers=0,pin_memory=True)
    feats=[];yi=[];yp=[];pid=[];iid=[];start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for n,(x,yimg,p,ypt,im) in enumerate(loader):
            feats.append(model.forward_features(x.cuda(non_blocking=True)).float().squeeze(1).cpu().numpy())
            yi.extend(yimg.tolist());yp.extend(ypt.tolist());pid.extend(p);iid.extend(im)
            if (n+1)%100==0:print('EXTERNAL_EXTRACT',n+1,'/',len(loader),flush=True)
    a=np.concatenate(feats).astype('float32')
    assert a.shape==(1498,1024) and np.isfinite(a).all()
    CACHE.mkdir(parents=True,exist_ok=True)
    target=CACHE/'mbrset_target_test.npz'
    np.savez_compressed(target,features=a,y_image=np.array(yi,dtype='int8'),
                        y_patient=np.array(yp,dtype='int8'),patient_id=np.array(pid),image_id=np.array(iid))
    meta={'cohort':'mBRSET_TARGET_TEST','images':1498,'patients':400,
          'seconds':round(time.perf_counter()-start,2),'peak_reserved_mib':round(torch.cuda.max_memory_reserved()/2**20,2),
          'weight_sha256':EXPECTED,'cache_sha256':sha(target),'load':loading}
    (OUT/'mbrset_feature_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    c=readcache('mbrset_target_test');rows=[];predrows=[]
    for seed in SEEDS:
        path=OUT/f'head_seed{seed}_best.pth';sel=OUT/f'head_seed{seed}_selection.json'
        assert sha(path)==json.loads(sel.read_text())['head_sha256']
        cp=torch.load(path,map_location='cpu',weights_only=True);h=Head();h.load_state_dict(cp['head'],strict=True)
        logits=predict(h,c);predrows.extend(patient_rows(logits,c,seed,'mBRSET_TARGET_TEST'))
        y,p=patient_score(logits,c);auc=float(roc_auc_score(y,p))
        rng=np.random.default_rng(seed+2000);pos=np.flatnonzero(y);neg=np.flatnonzero(1-y);bs=[]
        for _ in range(2000):
            ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
            bs.append(roc_auc_score(y[ix],p[ix]))
        lo,hi=np.quantile(bs,[.025,.975])
        rows.append({'seed':seed,'target_test_patients':len(y),'positive':int(y.sum()),'auroc':auc,
                     'ci_low':float(lo),'ci_high':float(hi),'bootstrap':2000,'head_sha256':sha(path)})
    with (OUT/'mbrset_target_test_results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (CACHE/'mbrset_target_test_patient_predictions_PRIVATE.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(predrows[0]));w.writeheader();w.writerows(predrows)
    print('EXTERNAL_TEST',json.dumps(rows),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--mode',choices=['smoke','extract','train','test','external'],required=True)
    mode=a.parse_args().mode
    {'smoke':gpu_smoke,'extract':extract,'train':train,'test':test,'external':external}[mode]()
