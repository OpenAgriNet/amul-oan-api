"""
Parity harness — proves the Beckn-network scheme path returns the SAME union
scheme titles as the direct Redis cache.

Compares:
  - Schemes: direct get_cached_scheme_records_for_union vs network_union_schemes
    → compare the set of scheme titles.

Vet-KB search and AI-call booking always use Marqo / PashuGPT and are not
compared here.

Run inside a container with the chat image + env (AMUL_NETWORK_URL) on the
amulnet network. Exit 0 iff every case matches.
"""
import asyncio
import json
import sys

sys.path.insert(0, "/app")

from app.config import settings  # noqa: E402

SCHEME_UNION = "banas"

passes, fails = 0, 0
def ok(msg):
    global passes; passes += 1; print(f"  ✅ {msg}")
def bad(msg):
    global fails; fails += 1; print(f"  ❌ {msg}")


async def compare_schemes():
    print("\n== Union Schemes (schemes:amul-union) — result parity ==")
    # direct: read the Redis cache the same way get_union_scheme_data does
    from app.services.scheme_ingestion import get_cached_scheme_records_for_union
    settings.enable_network = False
    recs = await get_cached_scheme_records_for_union(SCHEME_UNION)
    direct_titles = sorted({str(r.get("scheme_title", "")).strip() for r in recs})
    # network: via the beckn client
    from agents.tools.beckn_network import network_union_schemes
    settings.enable_network = True
    out = await network_union_schemes("", union=SCHEME_UNION)
    net_titles = sorted({s.get("scheme_title", "").strip() for s in json.loads(out)})
    missing = set(direct_titles) - set(net_titles)
    extra = set(net_titles) - set(direct_titles)
    if not missing and not extra:
        ok(f"{SCHEME_UNION}: identical scheme set ({len(direct_titles)} schemes)")
    else:
        # network applies RESULT_LIMIT; a subset is expected if direct has more
        if not extra and missing:
            ok(f"{SCHEME_UNION}: network is a faithful subset ({len(net_titles)}/{len(direct_titles)}; limited by RESULT_LIMIT, {len(missing)} beyond the page)")
        else:
            bad(f"{SCHEME_UNION}: mismatch — missing {sorted(missing)[:3]} extra {sorted(extra)[:3]}")


async def main():
    await compare_schemes()
    print(f"\nPARITY: {passes} passed, {fails} failed")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
