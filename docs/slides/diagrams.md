# Slide diagrams (Mermaid source)

The diagrams of `presentation.pptx`, drawn wide for 16:9 slides. `render_diagrams.js` renders every
block named with a `<!-- diagram: name -->` comment to `diagrams/<name>.png`. The guide
(`docs/guide.md`) has fuller versions of the same diagrams.

Classes: `a` for Approach A (blue), `b` for Approach B (orange), `shared` for steps both use.

<!-- diagram: logbert-train -->
```mermaid
flowchart LR
    seq["Normal window<br>#lt;dist#gt; 17 17 52 8 17"] --> mask["Hide half the keys<br>#lt;dist#gt; 17 #lt;mask#gt; 52 #lt;mask#gt; 17"]
    mask --> model["LogBERT<br>2-layer Transformer"]
    model --> guess["Guess the hidden keys<br>MLKP loss"]
    model --> dist["Pull the #lt;dist#gt; vector<br>towards a centre: VHM loss"]
    class seq,mask,model,guess,dist a
    classDef a fill:#e7f1fb,stroke:#2a78d6,color:#0b0b0b
```

<!-- diagram: logbert-detect -->
```mermaid
flowchart LR
    w["Test window"] --> enc["IDs → tokens<br>new → #lt;unk#gt;"]
    enc --> mask["Hide keys, rank<br>each true key"]
    mask --> miss["Missed if not<br>in the top g"]
    miss --> count{"More than r<br>missed?"}
    count -->|yes| anom["ANOMALOUS"]
    count -->|no| norm["normal"]
    class w,enc,mask,miss,count a
    class anom alert
    class norm ok
    classDef a fill:#e7f1fb,stroke:#2a78d6,color:#0b0b0b
    classDef alert fill:#fdeee7,stroke:#eb6834,color:#0b0b0b
    classDef ok fill:#f3f5f8,stroke:#c3c2b7,color:#0b0b0b
```

<!-- diagram: logrules -->
```mermaid
flowchart LR
    pool["442 labelled<br>templates"] --> induce["LLM writes<br>rules"]
    induce --> cand["160<br>candidates"]
    cand --> verify["Keep the 99<br>never wrong<br>on training"]
    verify --> rank["Rank by<br>coverage:<br>top 30"]
    rank --> b3["B3: LLM<br>applies them"]
    rank --> b4["B4: keyword<br>matching"]
    class pool,induce,cand,verify,rank,b3,b4 b
    classDef b fill:#fdeee7,stroke:#eb6834,color:#0b0b0b
```

<!-- diagram: run-flow -->
```mermaid
flowchart LR
    pre["preprocess"] --> trainA["train<br>LogBERT"]
    trainA --> evalA["tune g, r<br>score test"]
    pre --> ind["induce<br>rules"]
    ind --> runB["run<br>B1-B4"]
    evalA --> cmp["compare on the<br>same windows"]
    runB --> cmp
    cmp --> an["analyze"]
    an --> res["results/"]
    class pre,cmp,an,res shared
    class trainA,evalA a
    class ind,runB b
    classDef shared fill:#f3f5f8,stroke:#52514e,color:#0b0b0b
    classDef a fill:#e7f1fb,stroke:#2a78d6,color:#0b0b0b
    classDef b fill:#fdeee7,stroke:#eb6834,color:#0b0b0b
```
