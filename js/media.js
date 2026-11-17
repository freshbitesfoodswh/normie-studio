// Recording, audio cleanup, takes storage
// H.264+AAC first (iPhone/Mac/Windows), then H.264+Opus (Linux Chrome), then whatever MP4 the browser has
const MP4=['video/mp4;codecs=avc1.640028,mp4a.40.2','video/mp4;codecs=avc1.42E01F,mp4a.40.2','video/mp4;codecs=avc1,mp4a.40.2','video/mp4;codecs=avc1,opus','video/mp4'];
export function pickMime(){
  const c=[...MP4,'video/webm;codecs=vp9,opus','video/webm;codecs=vp8,opus','video/webm'];
  if(typeof MediaRecorder==='undefined') return null;
  for(const m of c) if(MediaRecorder.isTypeSupported(m)) return m; return '';
}
export function pickMimeFor(fmt){
  if(typeof MediaRecorder==='undefined') return null;
  const mp4=MP4;
  const webm=['video/webm;codecs=vp9,opus','video/webm;codecs=vp8,opus','video/webm'];
  for(const m of (fmt==='mp4'?mp4:webm)) if(MediaRecorder.isTypeSupported(m)) return m; return null;
}
export const extFor=m=>m&&m.includes('mp4')?'mp4':'webm';

const GATE_SRC=`class Gate extends AudioWorkletProcessor{constructor(){super();this.env=0;this.g=1;this.hold=0;}
static get parameterDescriptors(){return[{name:'threshold',defaultValue:-48},{name:'floor',defaultValue:0.2}]}
process(inp,out,p){const i=inp[0],o=out[0];if(!i||!i.length)return true;const thr=Math.pow(10,p.threshold[0]/20),fl=p.floor[0];const n=i[0].length;
for(let s=0;s<n;s++){let m=0;for(let c=0;c<i.length;c++){const v=Math.abs(i[c][s]);if(v>m)m=v;}
this.env=m>this.env?m:this.env*0.9993; if(this.env>thr)this.hold=sampleRate*0.15; else if(this.hold>0)this.hold--;
const tg=this.hold>0?1:fl; this.g+=(tg-this.g)*(tg>this.g?0.02:0.0006);
for(let c=0;c<o.length;c++)o[c][s]=(i[c]||i[0])[s]*this.g;}return true;}}
registerProcessor('normie-gate',Gate);`;

let ctx=null, gateReady=null;
export function audioCtx(){ if(!ctx){ ctx=new (window.AudioContext||window.webkitAudioContext)(); } if(ctx.state==='suspended') ctx.resume(); return ctx; }
async function ensureGate(){ const c=audioCtx(); if(!gateReady){ gateReady=(async()=>{ try{ const url=URL.createObjectURL(new Blob([GATE_SRC],{type:'application/javascript'})); await c.audioWorklet.addModule(url); return true; }catch(e){ console.warn('gate unavailable',e); return false; } })(); } return gateReady; }

// Builds: element -> [clean chain] -> MediaStreamDestination (+ optional monitor)
const elSources=new WeakMap();
export async function elementAudio(el, clean){
  const c=audioCtx(); let src=elSources.get(el); if(!src){ src=c.createMediaElementSource(el); elSources.set(el,src); }
  try{ src.disconnect(); }catch{}
  const dest=c.createMediaStreamDestination(); const nodes=[];
  let last=src;
  if(clean){
    const hp=c.createBiquadFilter(); hp.type='highpass'; hp.frequency.value=85; hp.Q.value=0.7;
    const presence=c.createBiquadFilter(); presence.type='peaking'; presence.frequency.value=3200; presence.gain.value=2.5; presence.Q.value=0.9;
    const comp=c.createDynamicsCompressor(); comp.threshold.value=-24; comp.knee.value=12; comp.ratio.value=3.5; comp.attack.value=0.005; comp.release.value=0.2;
    const mk=c.createGain(); mk.gain.value=1.6;
    last.connect(hp); last=hp;
    if(await ensureGate()){ const g=new AudioWorkletNode(c,'normie-gate'); last.connect(g); last=g; nodes.push(g); }
    last.connect(presence); presence.connect(comp); comp.connect(mk); last=mk; nodes.push(hp,presence,comp,mk);
  }
  last.connect(dest);
  const monitor=c.createGain(); monitor.gain.value=1; last.connect(monitor); monitor.connect(c.destination);
  return {stream:dest.stream, monitor, src};
}

// --------- IndexedDB takes ---------
const DB='normie-studio', ST='takes';
function db(){ return new Promise((res,rej)=>{ const r=indexedDB.open(DB,1); r.onupgradeneeded=()=>r.result.createObjectStore(ST,{keyPath:'id'}); r.onsuccess=()=>res(r.result); r.onerror=()=>rej(r.error); }); }
export async function saveTake(t){ try{ const d=await db(); await new Promise((res,rej)=>{ const tx=d.transaction(ST,'readwrite'); tx.objectStore(ST).put(t); tx.oncomplete=res; tx.onerror=()=>rej(tx.error); }); }catch(e){ console.warn('save failed',e); } }
export async function loadTakes(){ try{ const d=await db(); return await new Promise((res,rej)=>{ const r=d.transaction(ST).objectStore(ST).getAll(); r.onsuccess=()=>res(r.result||[]); r.onerror=()=>rej(r.error); }); }catch(e){ return []; } }
export async function deleteTake(id){ try{ const d=await db(); await new Promise(res=>{ const tx=d.transaction(ST,'readwrite'); tx.objectStore(ST).delete(id); tx.oncomplete=res; }); }catch{} }
