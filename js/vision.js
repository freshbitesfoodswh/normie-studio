// Free MediaPipe models from Google's public bucket + jsDelivr CDN (no keys).
const VER='1.2.3';
const CDN=`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VER}`;
const FACE_MODEL='https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const SEG_MODEL='https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite';

export class Vision {
  constructor(){ this.face=null; this.seg=null; this.status='loading'; this.lastTs=0; this.ema=null; }
  async load(onStatus){
    const mp=await import(`${CDN}/vision_bundle.mjs`);
    const files=await mp.FilesetResolver.forVisionTasks(`${CDN}/wasm`);
    const mk=async(fn)=>{ try{ return await fn('GPU'); }catch(e){ console.warn('GPU delegate failed, using CPU',e); return await fn('CPU'); } };
    const [face,seg]=await Promise.allSettled([
      mk(d=>mp.FaceLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:FACE_MODEL,delegate:d},runningMode:'VIDEO',numFaces:1,
        minFaceDetectionConfidence:0.5,minFacePresenceConfidence:0.5,minTrackingConfidence:0.5})),
      mk(d=>mp.ImageSegmenter.createFromOptions(files,{baseOptions:{modelAssetPath:SEG_MODEL,delegate:d},runningMode:'VIDEO',outputConfidenceMasks:true,outputCategoryMask:false})),
    ]);
    this.face=face.status==='fulfilled'?face.value:null; this.seg=seg.status==='fulfilled'?seg.value:null;
    this.status=(this.face&&this.seg)?'ready':(this.face||this.seg)?'partial':'failed';
    onStatus&&onStatus(this.status); return this.status;
  }
  ts(){ let t=performance.now(); if(t<=this.lastTs) t=this.lastTs+0.01; this.lastTs=t; return t; }
  detectFace(src){ if(!this.face) return null; try{ const r=this.face.detectForVideo(src,this.ts()); return r.faceLandmarks&&r.faceLandmarks[0]||null; }catch(e){ console.warn(e); return null; } }
  segment(src){ if(!this.seg) return null; let out=null;
    try{ this.seg.segmentForVideo(src,this.ts(),res=>{ const m=res.confidenceMasks&&res.confidenceMasks[0]; if(!m) return;
        const f=m.getAsFloat32Array(), w=m.width, h=m.height;
        if(!this.ema||this.ema.length!==f.length){ this.ema=new Float32Array(f); this.u8=new Uint8Array(f.length); }
        const e=this.ema,u=this.u8; for(let i=0;i<f.length;i++){ e[i]=e[i]*0.45+f[i]*0.55; u[i]=e[i]*255; }
        out={u8:u,w,h}; }); }catch(e){ console.warn(e); }
    return out; }
}

// ---------- landmark smoothing (One Euro filter) ----------
class OneEuro{ constructor(minC=1.2,beta=0.6,dC=1.0){ this.minC=minC; this.beta=beta; this.dC=dC; this.x=null; this.dx=null; this.t=0; }
  a(c,dt){ const r=2*Math.PI*c*dt; return r/(r+1); }
  f(v,t){ if(this.x===null){ this.x=v.slice(); this.dx=new Float32Array(v.length); this.t=t; return this.x; }
    const dt=Math.max(1/120,Math.min(0.2,(t-this.t)/1000)); this.t=t;
    const ad=this.a(this.dC,dt);
    for(let i=0;i<v.length;i++){ const d=(v[i]-this.x[i])/dt; this.dx[i]+=ad*(d-this.dx[i]); }
    // speed computed from mean |dx| so the whole face moves coherently
    let sp=0; for(let i=0;i<v.length;i++) sp+=Math.abs(this.dx[i]); sp/=v.length;
    const al=this.a(this.minC+this.beta*sp,dt);
    for(let i=0;i<v.length;i++) this.x[i]+=al*(v[i]-this.x[i]);
    return this.x; } }

// ---------- geometry ----------
const LIP_OUT=[61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146];
const LIP_IN =[78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95];
const EYE_R=[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246];
const EYE_L=[263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466];
const UP_R=[33,246,161,160,159,158,157,173,133], LO_R=[33,7,163,144,145,153,154,155,133];
const UP_L=[263,466,388,387,386,385,384,398,362], LO_L=[263,249,390,373,374,380,381,382,362];
const BROW_R=[[70,46],[63,53],[105,52],[66,65],[107,55]], BROW_L=[[300,276],[293,283],[334,282],[296,295],[336,285]];
const OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
export const USED=new Set([...LIP_OUT,...LIP_IN,...EYE_R,...EYE_L,...OVAL,...BROW_R.flat(),...BROW_L.flat(),468,469,470,471,472,473,474,475,476,477,1,4,152,234,454,10]);

export class FaceTracker{
  constructor(){ this.filter=new OneEuro(); this.weight=0; this.last=null; this.lost=0; }
  // lm: normalized source landmarks; crop {off,scale}; W,H output size
  update(lm,crop,W,H,t){
    if(lm){ const v=new Float32Array(478*2); const n=Math.min(478,lm.length);
      for(let i=0;i<n;i++){ v[i*2]=(lm[i].x-crop.off[0])/crop.scale[0]; v[i*2+1]=(lm[i].y-crop.off[1])/crop.scale[1]; }
      if(this.last&&this.weight<0.05) this.filter=new OneEuro();
      this.last=this.filter.f(v,t); this.weight=Math.min(1,this.weight+0.25); this.lost=0; }
    else { this.lost++; if(this.lost>3) this.weight=Math.max(0,this.weight-0.15); }
    if(!this.last||this.weight<=0) return null;
    return geometry(this.last,W,H,this.weight);
  }
  reset(){ this.filter=new OneEuro(); this.weight=0; this.last=null; }
}

function geometry(P,W,H,weight){
  const asp=W/H; // work in "height units": x*asp, y
  const p=i=>[P[i*2],P[i*2+1]];
  const px=i=>[P[i*2]*asp,P[i*2+1]];
  const fromH=q=>[q[0]/asp,q[1]];
  const dist=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
  const tri=[]; const push=(...pts)=>{ for(const q of pts) tri.push(q[0],q[1]); };
  const fan=(idx,out)=>{ let cx=0,cy=0; for(const i of idx){ cx+=P[i*2]; cy+=P[i*2+1]; } cx/=idx.length; cy/=idx.length;
    for(let k=0;k<idx.length;k++){ const a=p(idx[k]),b=p(idx[(k+1)%idx.length]); out.push(cx,cy,a[0],a[1],b[0],b[1]); } };
  const strip=(A,B,out,closed)=>{ const n=A.length; const m=closed?n:n-1; for(let k=0;k<m;k++){ const a0=A[k],a1=A[(k+1)%n],b0=B[k],b1=B[(k+1)%n];
      out.push(a0[0],a0[1],a1[0],a1[1],b0[0],b0[1], a1[0],a1[1],b1[0],b1[1],b0[0],b0[1]); } };
  // lips: ring between outer and inner contour
  const lips=[]; strip(LIP_OUT.map(p),LIP_IN.map(p),lips,true);
  // eyes
  const eyes=[]; fan(EYE_R,eyes); fan(EYE_L,eyes);
  // brows (strip upper/lower)
  const brows=[]; strip(BROW_R.map(x=>p(x[0])),BROW_R.map(x=>p(x[1])),brows,false); strip(BROW_L.map(x=>p(x[0])),BROW_L.map(x=>p(x[1])),brows,false);
  // under-eye and lash line, computed in height units
  const under=[], lash=[];
  const eyeC=[],eyeR=[],irisC=[],irisR=[],gazeD=[];
  for(const [UP,LO,IRC] of [[UP_R,LO_R,468],[UP_L,LO_L,473]]){
    const o=px(UP[0]), inn=px(UP[UP.length-1]); const ew=dist(o,inn);
    const ax=[(inn[0]-o[0])/ew,(inn[1]-o[1])/ew]; let nrm=[-ax[1],ax[0]];
    const upc=UP.map(px).reduce((s,q)=>[s[0]+q[0],s[1]+q[1]],[0,0]).map(v=>v/UP.length);
    const loc=LO.map(px).reduce((s,q)=>[s[0]+q[0],s[1]+q[1]],[0,0]).map(v=>v/LO.length);
    if((loc[0]-upc[0])*nrm[0]+(loc[1]-upc[1])*nrm[1]<0) nrm=[-nrm[0],-nrm[1]]; // nrm points down (toward lower lid)
    const eh=Math.max(ew*0.12,dist(upc,loc)*2);
    // lash: upper lid outward (up), thicker at outer corner, small flick
    const A=[],B=[]; UP.forEach((i,k)=>{ const q=px(i); const t=ew*(0.075-0.05*k/(UP.length-1)); A.push(fromH([q[0]+nrm[0]*ew*0.012,q[1]+nrm[1]*ew*0.012])); B.push(fromH([q[0]-nrm[0]*t,q[1]-nrm[1]*t])); });
    const oq=px(UP[0]); const flick=fromH([oq[0]-ax[0]*ew*0.12-nrm[0]*ew*0.06,oq[1]-ax[1]*ew*0.12-nrm[1]*ew*0.06]);
    strip(A,B,lash,false); lash.push(A[0][0],A[0][1],B[0][0],B[0][1],flick[0],flick[1]);
    // under-eye crescent below lower lid
    const C=[],D=[]; LO.forEach((i,k)=>{ const q=px(i); const s=Math.sin(Math.PI*k/(LO.length-1)); const t0=ew*0.06, t1=ew*(0.10+0.32*s);
      C.push(fromH([q[0]+nrm[0]*t0,q[1]+nrm[1]*t0])); D.push(fromH([q[0]+nrm[0]*t1+ax[0]*0,q[1]+nrm[1]*t1])); });
    strip(C,D,under,false);
    const mid=[(o[0]+inn[0])/2,(o[1]+inn[1])/2];
    eyeC.push(mid[0]/asp,mid[1]); eyeR.push(ew*1.05);
    const ic=px(IRC); let r=0; for(let k=1;k<=4;k++) r+=dist(ic,px(IRC+k)); r/=4;
    irisC.push(ic[0]/asp,ic[1]); irisR.push(r*1.05);
    // gaze target: centre of the eye opening, nudged up a touch (lens is above the screen)
    let gx=mid[0]-ic[0], gy=(mid[1]-r*0.12)-ic[1]; const gm=Math.hypot(gx,gy), lim=r*0.7; if(gm>lim){ gx*=lim/gm; gy*=lim/gm; }
    gazeD.push(gx/asp,gy);
  }
  const oval=[]; fan(OVAL,oval);
  // lips warp + gloss
  const lo=LIP_OUT.map(px); let lx0=1e9,lx1=-1e9,ly0=1e9,ly1=-1e9; for(const q of lo){ lx0=Math.min(lx0,q[0]); lx1=Math.max(lx1,q[0]); ly0=Math.min(ly0,q[1]); ly1=Math.max(ly1,q[1]); }
  const lw=lx1-lx0, lh=ly1-ly0;
  const lipC=[(lx0+lx1)/2/asp,(ly0+ly1)/2];
  const lowMid=[(px(14)[0]+px(17)[0])/2,(px(14)[1]*0.55+px(17)[1]*0.45)];
  const upMid=[(px(13)[0]+px(0)[0])/2,(px(13)[1]*0.5+px(0)[1]*0.5)];
  let fx0=1e9,fx1=-1e9; for(const i of OVAL){ fx0=Math.min(fx0,P[i*2]); fx1=Math.max(fx1,P[i*2]); }
  const e0=[eyeC[0]*asp,eyeC[1]], e1=[eyeC[2]*asp,eyeC[3]];
  const roll=Math.atan2(e1[1]-e0[1],e1[0]-e0[0])*(e1[0]>=e0[0]?1:1);
  const ch=p(152); const faceWh=dist(px(234),px(454)), faceH=dist(px(10),px(152));
  return { weight, roll:(e1[0]<e0[0]?roll-Math.PI*Math.sign(roll):roll), chin:ch, faceWh, faceH,
    faceW:fx1-fx0,
    maskA:[{tris:new Float32Array(lips),ch:[1,0,0,0]},{tris:new Float32Array(eyes),ch:[0,1,0,0]},{tris:new Float32Array(under),ch:[0,0,1,0]},{tris:new Float32Array(brows),ch:[0,0,0,1]}],
    maskB:[{tris:new Float32Array(lash),ch:[1,0,0,0]},{tris:new Float32Array(oval),ch:[0,0,1,0]}],
    eyeC,eyeR,irisC,irisR,gazeD,
    lipC,lipR:[lw*0.62,Math.max(lh*1.05,lw*0.3)],
    glossC:[lowMid[0]/asp,lowMid[1],upMid[0]/asp,upMid[1]], glossR:[lw*0.16,Math.max(lh*0.12,lw*0.05)],
  };
}
