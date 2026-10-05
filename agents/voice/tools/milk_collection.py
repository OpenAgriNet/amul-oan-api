"""Tool for fetching farmer milk collection and deduction details."""
import os

from pydantic import ValidationError
from pydantic_ai import RunContext

from agents.deps import FarmerAccount, FarmerContext
from app.voice.models.milk_collection import FarmerMilkCollectionRequestModel
from agents.voice.tools.farmer_animal_backends import get_farmer_milk_collection_details_api
from helpers.utils import get_logger

logger = get_logger(__name__)


def _num(value) -> str:
    """Render a numeric field compactly, dropping a trailing .0 (e.g. 2.0 -> '2')."""
    if value is None:
        return "unknown"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


_SHIFT_LABELS = {"M": "morning", "E": "evening"}


def _format_milk_collection_summary(response) -> str:
    """Deterministic plain-text summary for the voice agent.

    The voice agent runs on a small OSS model and speaks its answer aloud, so
    it gets a flat, labelled, per-record list (one record per line) instead of
    raw JSON or a markdown table. Each field is named so the model cannot
    confuse quantity with fat/SNF/amount when several records are present.
    """
    lines: list[str] = []

    if response.milk:
        lines.append(f"Milk collection records ({len(response.milk)}):")
        for i, r in enumerate(response.milk, 1):
            shift = _SHIFT_LABELS.get((r.shift or "").upper(), r.shift or "unknown")
            lines.append(
                f"  {i}. Date {r.date or 'unknown'}, {shift} shift: "
                f"quantity {_num(r.qty)} liters, "
                f"fat {_num(r.fat)}, SNF {_num(r.snf)}, "
                f"amount {_num(r.amount)} rupees."
            )
    else:
        lines.append("No milk collection records for the selected date range.")

    if response.deduction:
        lines.append(f"Deduction records ({len(response.deduction)}):")
        for i, d in enumerate(response.deduction, 1):
            lines.append(
                f"  {i}. Date {d.date or 'unknown'}, "
                f"{d.account_name or 'account'}: amount {_num(d.amount)} rupees."
            )
    else:
        lines.append("No deductions for the selected date range.")

    return "\n".join(lines)


async def _fetch_one_account(
    account: FarmerAccount,
    fromdate: str,
    todate: str,
    token: str,
):
    """Fetch milk/deduction for one account. Returns the response, or None on failure.

    Dates are validated once by the caller before fan-out, so a per-account
    failure here is always an upstream/API issue, not a date problem.
    """
    request = FarmerMilkCollectionRequestModel(
        unionCode=account.union_code or "",
        societyCode=account.society_code or "",
        farmerCode=account.farmer_code or "",
        fromdate=fromdate,
        todate=todate,
    )
    response = await get_farmer_milk_collection_details_api(request, token)
    logger.info(
        "Milk collection lookup: union=%s society=%s farmer=%s from=%s to=%s ok=%s milk=%s ded=%s",
        account.union_code, account.society_code, account.farmer_code, fromdate, todate,
        response is not None,
        len(response.milk) if response else 0,
        len(response.deduction) if response else 0,
    )
    return response


def _account_label(account: FarmerAccount, multi: bool) -> str:
    """Header for an account's section, only shown when fanning out over >1 account."""
    if not multi:
        return ""
    parts = []
    if account.society_name:
        parts.append(account.society_name)
    if account.farmer_code:
        parts.append(f"farmer code {account.farmer_code}")
    label = ", ".join(parts) if parts else f"account {account.farmer_code or '?'}"
    return f"Account — {label}:"


async def get_farmer_milk_collection_details(
    ctx: RunContext[FarmerContext],
    union_code: str,
    society_code: str,
    farmer_code: str,
    fromdate: str,
    todate: str,
) -> str:
    """
    Fetch milk collection and deduction details for the signed-in farmer.

    A single mobile number can have more than one account (for example a
    separate cow account and buffalo account). This tool automatically looks
    up every account on the caller's mobile and reports them together, so you
    do not need to pick one. The codes you pass are ignored whenever the
    caller's accounts are known, which is every turn this tool is offered.

    Args:
        ctx: The run context (automatically provided).
        union_code: Union code from farmer context.
        society_code: Society code from farmer context.
        farmer_code: Farmer code from farmer context.
        fromdate: Start date in YYYY-MM-DD format (ISO).
        todate: End date in YYYY-MM-DD format (ISO).

    Returns:
        str: Formatted milk collection and deduction details across all of the
             farmer's accounts, or a clear failure message.
    """
    # Accounts come from context only. This used to fall back to the model's own
    # codes when context had none — but context is empty precisely when identity
    # is unresolved, so the fallback ran on invented codes every time. 117 of
    # 1,148 voice lookups (10.2%) took it, and 96 of those returned "fetched
    # successfully - no milk collection records", telling registered farmers they
    # had none. That is worse than an error: the lookup succeeds and the wrong
    # answer is spoken with confidence. The tool is now withheld on those turns
    # (agents.voice.services.farmer_identity); refusing here is the backstop. See #282.
    accounts = list(ctx.deps.farmer_accounts) if ctx.deps and ctx.deps.farmer_accounts else []
    if not accounts:
        logger.warning(
            "Milk collection blocked: no farmer accounts in context (llm_codes=%s/%s/%s)",
            union_code, society_code, farmer_code,
        )
        return (
            "Milk collection lookup failed. The farmer account details are not available."
        )
    logger.info(
        "Milk collection tool invoked: accounts=%s from=%s to=%s (llm_codes=%s/%s/%s)",
        len(accounts), fromdate, todate, union_code, society_code, farmer_code,
    )
    return await fetch_milk_summary_for_accounts(accounts, fromdate, todate)


async def fetch_milk_summary_for_accounts(
    accounts: list[FarmerAccount],
    fromdate: str,
    todate: str,
) -> str:
    """Fan out over every account and render one plain-text summary.

    Split out of the tool body so callers without a ``RunContext`` can reuse the
    exact same fetch + formatting — notably the outbound-call prefetch, which
    warms this summary while the caller is still hearing the intro line (see
    ``app.services.outbound``). Any change to the wording the agent reads must
    therefore stay in here, not in the tool wrapper.
    """
    token = os.getenv("PASHUGPT_TOKEN")
    if not token:
        logger.error("PASHUGPT_TOKEN is not set")
        return "Milk collection lookup failed. Service is not configured."

    if not accounts:
        return "Milk collection lookup failed. No farmer account is available."

    # Validate the date range once — it is the same for every account, so a bad
    # date is a single clear failure rather than a per-account error.
    try:
        FarmerMilkCollectionRequestModel(
            unionCode=accounts[0].union_code or "",
            societyCode=accounts[0].society_code or "",
            farmerCode=accounts[0].farmer_code or "",
            fromdate=fromdate,
            todate=todate,
        ).validate_date_range()
    except (ValidationError, ValueError) as e:
        logger.info("Milk collection date validation failed: from=%s to=%s error=%s", fromdate, todate, e)
        return f"Milk collection lookup failed. {e}"

    multi = len(accounts) > 1
    sections: list[str] = []
    any_success = False
    total_milk = 0

    for account in accounts:
        result = await _fetch_one_account(account, fromdate, todate, token)
        if result is None:
            sections.append(
                (_account_label(account, multi) + "\n" if multi else "")
                + "Unable to fetch milk collection details for this account right now."
            )
            continue

        any_success = True
        total_milk += len(result.milk)
        summary = _format_milk_collection_summary(result)
        label = _account_label(account, multi)
        sections.append(f"{label}\n{summary}" if label else summary)

    if not any_success:
        return "Milk collection lookup failed. Unable to fetch details at the moment."

    body = "\n\n".join(sections)
    logger.info(
        "Milk collection aggregate: accounts=%s any_success=%s total_milk_records=%s",
        len(accounts), any_success, total_milk,
    )
    return f"Milk collection details fetched successfully:\n\n{body}"
