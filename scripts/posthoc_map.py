"""Apply the deterministic forbidden map to bare output post-hoc.

Tests whether R1 is actually a blocker, or just an unported layer. The map is
Gujarati->Gujarati, so unlike proper-noun pinning and glossary injection it does
NOT gate on an English source -- meaning it ports to bare mode unchanged.
"""
import csv,json,re,sys
csv.field_size_limit(sys.maxsize)
pol=json.load(open(sys.argv[1],encoding="utf-8"))
forb={k.strip():v.strip() for k,v in pol.get("forbidden",{}).items() if k.strip() and v.strip()}
pairs=sorted(forb.items(),key=lambda kv:len(kv[0]),reverse=True)  # longest first
def apply_map(t):
    for bad,good in pairs:
        t=t.replace(bad,good)
    return t
for path in sys.argv[2:]:
    rows=list(csv.DictReader(open(path,encoding="utf-8")))
    before=after=0; fixed=0
    for r in rows:
        a=r.get("answer","")
        if not a.strip(): continue
        b=any(k in a for k in forb)
        a2=apply_map(a)
        aft=any(k in a2 for k in forb)
        before+=b; after+=aft
        if b and not aft: fixed+=1
    print(f"{path.split('/')[-1]:26} violating_before={before:4d}  after_map={after:4d}  fixed={fixed:4d}  ({100*fixed/max(before,1):.1f}% of violations)")
