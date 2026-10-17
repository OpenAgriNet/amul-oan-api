import csv,re,sys
csv.field_size_limit(sys.maxsize)
FEM=[(re.compile(r"(^|[,।.!?]\s+)\s*હું(?P<body>[^.!?\n]{0,80}?)શકું\s+ન(?:થી|હીં|હિ)(?=\s|[,।.!?]|$)"),r"\1હું\g<body>શકતી નથી"),
     (re.compile(r"(^|[,।.!?]\s+)\s*હું(?P<body>[^.!?\n]{0,80}?)શકું\s+છું(?=\s|[,।.!?]|$)"),r"\1હું\g<body>શકતી છું"),
     (re.compile(r"(^|[,।.!?]\s+)\s*હું(?P<body>[^.!?\n]{0,80}?)કરું(?=\s|[,।.!?]|$)"),r"\1હું\g<body>કરૂં"),
     (re.compile(r"(^|[,।.!?]\s+)\s*હું(?P<body>[^.!?\n]{0,80}?)આવું\s+છું(?=\s|[,।.!?]|$)"),r"\1હું\g<body>આવી છું")]
MASC=re.compile(r"હું[^.।!?]{0,40}(શકું|કરું|આવું|શકતો|કરતો|જાણતો)")
def femfix(t):
    for p,r in FEM: t=p.sub(r,t)
    return t
for path in sys.argv[1:]:
    rows=[r for r in csv.DictReader(open(path,encoding="utf-8")) if (r.get("answer") or "").strip()]
    masc=[r for r in rows if MASC.search(r["answer"])]
    fixed=[r for r in masc if not MASC.search(femfix(r["answer"]))]
    resid=[r for r in masc if MASC.search(femfix(r["answer"]))]
    print(f"{path.split('/')[-1]:24} masc={len(masc):3d}  fixed_by_prod_regex={len(fixed):3d}  RESIDUAL={len(resid):3d}  coverage={100*len(fixed)/max(len(masc),1):.0f}%")
    pats={}
    for r in resid:
        m=MASC.search(femfix(r["answer"]))
        if m: pats[m.group(0)[:44]]=pats.get(m.group(0)[:44],0)+1
    for k,v in sorted(pats.items(),key=lambda x:-x[1])[:5]:
        print(f"     uncovered x{v}: {k}")
