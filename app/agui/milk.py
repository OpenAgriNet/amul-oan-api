"""Present existing provider records without passing values through an LLM."""
from collections import defaultdict
from datetime import datetime

from agents.tools.milk_collection import milk_totals
from app.agui.models import MilkResult

TEXT = {
    "en": ("Choose your account", "Choose the milk record dates", "Milk records", "Milk records are temporarily unavailable. Please request them again."),
    "gu": ("તમારું ખાતું પસંદ કરો", "દૂધના રેકોર્ડની તારીખો પસંદ કરો", "દૂધના રેકોર્ડ", "દૂધના રેકોર્ડ હાલમાં ઉપલબ્ધ નથી. કૃપા કરીને ફરી વિનંતી કરો."),
    "hi": ("अपना खाता चुनें", "दूध के रिकॉर्ड की तारीखें चुनें", "दूध के रिकॉर्ड", "दूध के रिकॉर्ड अभी उपलब्ध नहीं हैं। कृपया फिर अनुरोध करें।"),
    "mr": ("तुमचे खाते निवडा", "दुधाच्या नोंदींच्या तारखा निवडा", "दुधाच्या नोंदी", "दुधाच्या नोंदी सध्या उपलब्ध नाहीत. कृपया पुन्हा विनंती करा."),
}


def labels(language: str) -> tuple[str, ...]:
    return TEXT.get(language.lower(), TEXT["en"])


def account_identity(account) -> tuple[str, str, str]:
    return account.union_code, account.society_code, account.farmer_code


def account_label(account) -> str:
    return " · ".join(x for x in (account.farmer_name, account.society_name,
                                  f"{account.union_code}/{account.society_code}/{account.farmer_code}") if x)


def milk_result(account: dict, period: dict, response) -> dict:
    daily = defaultdict(list)
    warnings = set()
    for row in response.milk:
        day = None
        for fmt in ("%Y-%m-%d", "%d-%m-%Y"):
            try:
                day = datetime.strptime(row.date or "", fmt).date().isoformat()
                break
            except ValueError:
                pass
        if day is None:
            warnings.add("unplottable_dates")
        else:
            daily[day].append(row.qty)
        if row.qty is None:
            warnings.add("missing_quantities")
    summary = None
    if response.milk:
        days, litres, amount, fat, snf = milk_totals(response.milk)
        summary = dict(days_with_milk=days, total_quantity_litres=round(litres, 2),
                       total_amount_rupees=round(amount, 2),
                       weighted_fat=round(fat, 2) if fat is not None else None,
                       weighted_snf=round(snf, 2) if snf is not None else None)
    result = MilkResult(
        status="ok" if response.milk or response.deduction else "no_records",
        account=account, period=period, summary=summary,
        milk=[r.model_dump() for r in response.milk],
        deductions=[r.model_dump() for r in response.deduction],
        daily_quantity=[dict(date=day, quantity_litres=(sum(v for v in values if v is not None)
                                                       if any(v is not None for v in values) else None))
                        for day, values in sorted(daily.items())],
        warnings=sorted(warnings),
    )
    return result.model_dump(mode="json")
