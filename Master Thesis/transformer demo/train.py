"""Train actual causal next-token reconstruction and write TensorBoard metrics.

Default rounds are sample-balanced window sampling, not complete corpus epochs.
--full-pass visits every body target exactly once per round (much slower).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from model import PrefixTransformer

ROOT=Path(__file__).resolve().parent
GROUP_NAMES={0:"topology",1:"vertex",2:"edge_bbox",3:"surface"}


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2))


def load_corpus(names,vocab):
    corpus={}
    for name in names:
        raw=json.loads((ROOT/"data/sequences"/f"{name}.json").read_text())
        variants={}
        for kind,seq in raw.items():
            variants[kind]=dict(ids=torch.tensor([vocab[t] for t in seq["tokens"]],dtype=torch.long),
                                groups=torch.tensor(seq["groups"]),prefix_length=seq["prefix_length"])
        corpus[name]=variants
    return corpus


def window(seq,start,target_tokens,history,device):
    p=seq["prefix_length"]; end=min(len(seq["ids"]),start+target_tokens)
    left=max(p,start-1-history)
    body=seq["ids"][left:end-1].to(device)
    y=seq["ids"][left+1:end].to(device)
    positions=torch.arange(left,end-1,device=device)
    loss_mask=torch.arange(left+1,end,device=device)>=start
    return seq["ids"][:p].to(device),body,positions,y,loss_mask,seq["groups"][left+1:end].to(device)


def stats(logits,y,mask,groups):
    losses=F.cross_entropy(logits,y,reduction="none")
    out={"loss":float(losses[mask].sum().detach()),"count":int(mask.sum()),
         "correct":int(((logits.argmax(-1)==y)&mask).sum())}
    for g,name in GROUP_NAMES.items():
        selected=mask&(groups==g)
        out[name+"_sum"]=float(losses[selected].sum().detach()); out[name+"_n"]=int(selected.sum())
    return losses[mask].mean(),out


def aggregate(rows):
    sums={k:sum(r[k] for r in rows) for k in rows[0]}
    out={"loss":sums["loss"]/sums["count"],"accuracy":sums["correct"]/sums["count"],"target_tokens":sums["count"]}
    for name in GROUP_NAMES.values():
        if sums[name+"_n"]: out["loss_"+name]=sums[name+"_sum"]/sums[name+"_n"]
    return out


def validation_windows(corpus,names,target_tokens):
    result=[]
    for name in names:
        s=corpus[name]["conditional"]; p=s["prefix_length"]; n=len(s["ids"])
        starts={p+1,max(p+1,(p+n)//2),max(p+1,n-target_tokens)}
        for g in (1,2,3):
            idx=torch.where(s["groups"][p+1:]==g)[0]
            if len(idx): starts.add(p+1+int(idx[len(idx)//2]))
        result.extend((name,start) for start in sorted(starts))
    return result


@torch.no_grad()
def evaluate(model,corpus,jobs,args):
    model.eval(); rows=[]
    for name,start in jobs:
        pref,body,pos,y,mask,groups=window(corpus[name]["conditional"],start,args.target_tokens,args.history,args.device)
        _,r=stats(model(pref,body,pos),y,mask,groups); rows.append(r)
    return aggregate(rows)


@torch.no_grad()
def reconstructions(model,corpus,names,vocab,args,outdir):
    model.eval(); results=[]
    for name in names:
        seq=corpus[name]["conditional"]; start=seq["prefix_length"]+1
        pref,body,pos,y,mask,groups=window(seq,start,128,args.history,args.device)
        pred=model(pref,body,pos).argmax(-1)
        results.append(dict(sample=name,mode="teacher_forced_one_step_predictions_NOT_free_running",
                            accuracy=float((pred[mask]==y[mask]).float().mean()),
                            target=[vocab[i] for i in y[mask].tolist()],prediction=[vocab[i] for i in pred[mask].tolist()]))
    dump(outdir/"teacher_forced_reconstruction.json",results)
    # Honest free-running continuation, without hidden access to target tokens.
    name=names[-1]; seq=corpus[name]["conditional"]; p=seq["prefix_length"]
    prefix=seq["ids"][:p].to(args.device); body=[int(seq["ids"][p])]; generated=[]
    for _ in range(96):
        left=max(0,len(body)-args.history-args.target_tokens)
        logits=model(prefix,torch.tensor(body[left:],device=args.device),torch.arange(p+left,p+len(body),device=args.device))
        token=int(logits[-1].argmax()); body.append(token); generated.append(vocab[token])
        if vocab[token]=="<EOS>": break
    dump(outdir/"free_running_continuation.json",dict(sample=name,mode="greedy_no_grammar_constraints",tokens=generated,
         warning="A short CPU run does not establish valid B-Rep generation; no STEP reconstruction is claimed."))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds",type=int,default=3)
    ap.add_argument("--windows-per-sample",type=int,default=2)
    ap.add_argument("--target-tokens",type=int,default=256)
    ap.add_argument("--history",type=int,default=128)
    ap.add_argument("--dim",type=int,default=64); ap.add_argument("--layers",type=int,default=2)
    ap.add_argument("--heads",type=int,default=4); ap.add_argument("--prefix-window",type=int,default=128)
    ap.add_argument("--dropout",type=float,default=0.1); ap.add_argument("--condition-dropout",type=float,default=0.1)
    ap.add_argument("--lr",type=float,default=0.001); ap.add_argument("--threads",type=int,default=4)
    ap.add_argument("--seed",type=int,default=42); ap.add_argument("--device",default="cpu")
    ap.add_argument("--run",default="demo_v1"); ap.add_argument("--full-pass",action="store_true")
    ap.add_argument("--resume",type=Path)
    args=ap.parse_args()
    torch.set_num_threads(args.threads); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng=random.Random(args.seed)
    vocabfile=ROOT/"data/vocabulary.json"; vdata=json.loads(vocabfile.read_text()); vocab=vdata["tokens"]
    vocab_sha=hashlib.sha256(vocabfile.read_bytes()).hexdigest()
    split=json.loads((ROOT/"data/split.json").read_text()); names=split["train"]+split["validation"]
    corpus=load_corpus(names,vdata["token_to_id"])
    config=dict(vocab_size=len(vocab),dim=args.dim,layers=args.layers,heads=args.heads,dropout=args.dropout,prefix_window=args.prefix_window)
    model=PrefixTransformer(**config).to(args.device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    outdir=ROOT/"checkpoints"/args.run; logdir=ROOT/"runs"/args.run
    if (outdir/"last.pt").exists() and not args.resume: raise FileExistsError("Run exists; choose --run NAME or --resume last.pt")
    outdir.mkdir(parents=True,exist_ok=True); start_round=0; step=0; history=[]; best=float("inf")
    if args.resume:
        ckpt=torch.load(args.resume,map_location=args.device)
        if ckpt["vocab_sha256"]!=vocab_sha: raise ValueError("Vocabulary changed")
        if ckpt["model_config"]!=config: raise ValueError("Resume model configuration differs from checkpoint")
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
        start_round=ckpt["round"]; step=ckpt["step"]; history=ckpt["history"]; best=ckpt["best_validation"]
        rng.setstate(ckpt["random_state"]); torch.set_rng_state(ckpt["torch_random_state"])
    writer=SummaryWriter(str(logdir),flush_secs=5,purge_step=step+1 if args.resume else None)
    run_config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    run_config.update(model=config,parameter_count=sum(p.numel() for p in model.parameters()),
       train_samples=len(split["train"]),validation_samples=len(split["validation"]),
       protocol="sample-balanced random target windows; all condition tokens retained; local causal prefix attention; no full-epoch claim" if not args.full_pass else "complete target-window pass",
       sources=["https://docs.pytorch.org/docs/stable/tensorboard.html","https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html"])
    dump(outdir/"config.json",run_config); dump(outdir/"vocabulary.json",vdata)
    writer.add_text("experiment/config",json.dumps(run_config,indent=2),step)
    jobs=validation_windows(corpus,split["validation"],args.target_tokens)
    baseline=evaluate(model,corpus,jobs,args)
    for k,v in baseline.items(): writer.add_scalar("validation/"+k,v,step)
    print(json.dumps(dict(stage="initial_validation",step=step,**baseline)),flush=True)
    started=time.monotonic(); sample_visits={n:0 for n in split["train"]}
    for rnd in range(start_round+1,start_round+args.rounds+1):
        schedule=[]
        for name in split["train"]:
            kind="unconditional" if rng.random()<args.condition_dropout else "conditional"
            seq=corpus[name][kind]; p=seq["prefix_length"]; n=len(seq["ids"])
            if args.full_pass: starts=range(p+1,n,args.target_tokens)
            else:
                starts=[rng.randrange(p+1,n) for _ in range(args.windows_per_sample)]
                # Alternate geometry-focused windows so bbox/point/surface events are actually trained.
                if rnd%2 and args.windows_per_sample>1:
                    idx=torch.where((seq["groups"]>=1)&(seq["groups"]<=3))[0]
                    if len(idx): starts[-1]=int(idx[rng.randrange(len(idx))])
            schedule.extend((name,kind,start) for start in starts)
        rng.shuffle(schedule); model.train(); rows=[]
        for name,kind,start in schedule:
            seq=corpus[name][kind]; pref,body,pos,y,mask,groups=window(seq,start,args.target_tokens,args.history,args.device)
            optimizer.zero_grad(set_to_none=True)
            loss,r=stats(model(pref,body,pos),y,mask,groups)
            if not torch.isfinite(loss): raise RuntimeError("Non-finite training loss")
            loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            step+=1; sample_visits[name]+=1; rows.append(r)
            writer.add_scalar("train/step_loss",float(loss.detach()),step)
            writer.add_scalar("train/step_accuracy",r["correct"]/r["count"],step)
            writer.add_scalar("optimizer/gradient_norm",float(grad),step)
            writer.add_scalar("optimizer/lr",optimizer.param_groups[0]["lr"],step)
            for group in GROUP_NAMES.values():
                if r[group+"_n"]: writer.add_scalar("train/"+group+"_loss",r[group+"_sum"]/r[group+"_n"],step)
            if step%30==0: print(json.dumps(dict(stage="training",round=rnd,step=step,loss=round(float(loss),4),seconds=round(time.monotonic()-started,1))),flush=True)
        train=aggregate(rows); val=evaluate(model,corpus,jobs,args)
        for k,v in train.items(): writer.add_scalar("round_train/"+k,v,step)
        for k,v in val.items(): writer.add_scalar("validation/"+k,v,step)
        record=dict(round=rnd,step=step,train=train,validation=val,seconds=time.monotonic()-started)
        history.append(record); improved=val["loss"]<best; best=min(best,val["loss"])
        ckpt=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),model_config=config,round=rnd,step=step,
                  history=history,best_validation=best,vocab_sha256=vocab_sha,
                  random_state=rng.getstate(),torch_random_state=torch.get_rng_state())
        torch.save(ckpt,outdir/"last.pt")
        if improved: torch.save(ckpt,outdir/"best.pt")
        dump(outdir/"metrics.json",dict(initial_validation=baseline,history=history,sample_visits=sample_visits,
            validation_protocol="fixed held-out windows, not full-sequence reconstruction accuracy"))
        writer.flush(); print(json.dumps(record),flush=True)
    reconstructions(model,corpus,[split["train"][0],split["validation"][0]],vocab,args,outdir)
    writer.close(); print(f"Finished. TensorBoard logdir: {logdir}",flush=True)


if __name__=="__main__": main()
