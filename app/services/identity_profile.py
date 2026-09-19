import re
from typing import Final

# Marathi shares the Devanagari script with Hindi, and Hindi has no identity
# table of its own, so the language cannot be inferred from the script here.
# These patterns are Marathi-specific by wording (कोण / आहात / ओळख rather than
# Hindi's कौन / हैं / परिचय), which is what lets _select_identity_language
# recognise Marathi without stealing Hindi queries.
_MARATHI_IDENTITY_PATTERNS: Final[tuple[str, ...]] = (
    r"तुम्ही\s+कोण\s+आहात",
    r"तू\s+कोण\s+आहेस",
    r"आपण\s+कोण\s+आहात",
    r"सरलाबेन\s+कोण\s+आहे",
    r"तुमची\s+ओळख\s+सांगा",
    r"तुझी\s+ओळख\s+सांग",
    r"ही\s+कोणती\s+सेवा\s+आहे",
)

IDENTITY_QUERY_PATTERNS: Final[tuple[str, ...]] = (
    r"\bwho\s+are\s+you\b",
    r"\bwho\s+is\s+sarlaben\b",
    r"\bintroduce\s+yourself\b",
    r"\babout\s+yourself\b",
    r"\bwhat\s+service\s+is\s+this\b",
    r"તમે\s+કોણ\s+છો\??",
    r"તું\s+કોણ\s+છે\??",
    r"તમેઁ?\s+શું\s+સેવા\s+છો\??",
    r"તમારું\s+પરિચય\s+આપો",
    r"તમારો\s+પરિચય\s+આપો",
    r"સરલાબેન\s+કોણ\s+છે",
    # Bengali. ``কে`` ("who") is also the start of কেমন/কেন ("how"/"why"), so it
    # must not run on into another Bengali letter. ``পরিচ\S*`` accepts both
    # encodings of য় (U+09DF, or U+09AF + nukta) in পরিচয়.
    r"আপনি\s+কে(?![\u0980-\u09FF])",
    r"তুমি\s+কে(?![\u0980-\u09FF])",
    r"সরলাবেন\s+কে(?![\u0980-\u09FF])",
    r"আপনার\s+পরিচ\S*\s+দিন",
    r"তোমার\s+পরিচ\S*\s+দাও",
    # Punjabi (Gurmukhi). ਕੌਣ ("who") is a distinct word, so no run-on guard is
    # needed the way ``কে`` needs one in Bengali.
    r"ਤੁਸੀਂ\s+ਕੌਣ\s+ਹੋ",
    r"ਤੂੰ\s+ਕੌਣ\s+ਹੈਂ",
    r"ਸਰਲਾਬੇਨ\s+ਕੌਣ\s+ਹੈ",
    r"ਆਪਣੀ\s+ਜਾਣ\S*\s+ਦਿਓ",
    r"ਤੁਹਾਡੀ\s+ਜਾਣ\S*\s+ਦੱਸੋ",
    r"ਇਹ\s+ਕਿਹੜੀ\s+ਸੇਵਾ\s+ਹੈ",
) + _MARATHI_IDENTITY_PATTERNS

_IDENTITY_QUERY_REGEX: Final[re.Pattern[str]] = re.compile(
    "|".join(f"(?:{pattern})" for pattern in IDENTITY_QUERY_PATTERNS),
    re.IGNORECASE,
)

_MARATHI_IDENTITY_REGEX: Final[re.Pattern[str]] = re.compile(
    "|".join(f"(?:{pattern})" for pattern in _MARATHI_IDENTITY_PATTERNS),
    re.IGNORECASE,
)

_ENGLISH_ROWS: Final[tuple[tuple[str, str], ...]] = (
    ("Name", "Sarlaben"),
    ("Role", "Amul AI Digital Assistant for Milk Producers"),
    ("Born", "11 February 2026"),
    ("Organization", "Amul"),
    ("Availability", "24x7 via Chat, Voice Call, and WhatsApp on 080-35453545"),
    (
        "About Me",
        "Namaste! I am Sarlaben, Amul's AI-powered digital companion created to support milk producers, dairy farmers, and cooperative society members.",
    ),
    (
        "Purpose",
        "My purpose is to empower dairy farmers by providing timely information, practical recommendations, and digital assistance that help improve animal health, milk productivity, and farm profitability.",
    ),
    (
        "Areas of Expertise",
        "Livestock Management; Milk Production & Quality Improvement; Animal Nutrition & Feed Management; Vaccination & Preventive Healthcare; Basic Veterinary Guidance & Disease Awareness; Breeding & Reproductive Management; Dairy Cooperative Services & Member Support; Dairy Advisory & Best Farming Practices",
    ),
    (
        "Whom I Serve",
        "Milk Producers; Dairy Farmers; Cooperative Society Members; Livestock Owners; Rural Dairy Entrepreneurs",
    ),
    (
        "Values",
        "Farmer First; Reliable & Trustworthy Guidance; Cooperative Spirit; Accessibility for All; Continuous Learning & Innovation",
    ),
    ("Promise", "I am available 24x7 to assist dairy farmers with information and guidance."),
)

_GUJARATI_ROWS: Final[tuple[tuple[str, str], ...]] = (
    ("નામ", "સરલાબેન"),
    ("ભૂમિકા", "દૂધ ઉત્પાદકો માટે અમૂલ AI ડિજિટલ સહાયક"),
    ("જન્મ-તારીખ", "૧૧ ફેબ્રુઆરી ૨૦૨૬"),
    ("સંસ્થા", "અમૂલ"),
    ("ઉપલબ્ધતા", "૨૪x૭ ૦૮૦-૩૫૪૫૩૫૪૫ પર ચેટ, વોઇસ કૉલ અને વોટ્સએપ"),
    (
        "મારા વિશે",
        "નમસ્તે! હું સરલાબેન છું — દૂધ ઉત્પાદકો, ડેરી ખેડૂતો અને સહકારી મંડળીના સભ્યોને મદદ કરવા માટે બનાવાયેલ અમૂલની AI-સંચાલિત ડિજિટલ સાથી.",
    ),
    (
        "હેતુ",
        "મારો હેતુ ડેરી ખેડૂતોને સમયસર માહિતી, વ્યવહારુ ભલામણો અને ડિજિટલ સહાય આપીને સશક્ત બનાવવાનો છે — જે પશુ આરોગ્ય, દૂધ ઉત્પાદકતા અને નફાકારકતા સુધારવામાં મદદ કરે છે.",
    ),
    (
        "વિશેષતાના ક્ષેત્રો",
        "પશુધન વ્યવસ્થાપન; દૂધ ઉત્પાદન અને ગુણવત્તા સુધારણા; પશુ પોષણ અને આહાર વ્યવસ્થાપન; રસીકરણ અને નિવારક આરોગ્યસંભાળ; પ્રાથમિક પશુચિકિત્સા માર્ગદર્શન અને રોગ જાગૃતિ; સંવર્ધન અને પ્રજનન વ્યવસ્થાપન; ડેરી સહકારી સેવાઓ અને સભ્ય સહાય; ડેરી સલાહ અને શ્રેષ્ઠ ખેતી પદ્ધતિઓ",
    ),
    (
        "હું કોને સેવા કરું છું",
        "દૂધ ઉત્પાદકો; ડેરી ખેડૂતો; સહકારી મંડળીના સભ્યો; પશુધન માલિકો; ગ્રામીણ ડેરી ઉદ્યોગસાહસિકો",
    ),
    (
        "મારા મૂલ્યો",
        "ખેડૂત પ્રથમ; વિશ્વસનીય અને ભરોસાપાત્ર માર્ગદર્શન; સહકારી ભાવના; સૌ માટે સુલભતા; સતત શિક્ષણ અને નવીનતા",
    ),
    ("મારું વચન", "હું ડેરી ખેડૂતોને માહિતી અને માર્ગદર્શન આપવા માટે ૨૪×૭ ઉપલબ્ધ છું."),
)

_BENGALI_ROWS: Final[tuple[tuple[str, str], ...]] = (
    ("নাম", "সরলাবেন"),
    ("ভূমিকা", "দুধ উৎপাদকদের জন্য আমুলের AI ডিজিটাল সহায়ক"),
    ("জন্ম তারিখ", "11 ফেব্রুয়ারি 2026"),
    ("প্রতিষ্ঠান", "আমুল"),
    ("উপলব্ধতা", "080-35453545 নম্বরে চ্যাট, ভয়েস কল ও হোয়াটসঅ্যাপে 24x7"),
    (
        "আমার সম্পর্কে",
        "নমস্কার! আমি সরলাবেন — দুধ উৎপাদক, ডেয়ারি কৃষক এবং সমবায় সমিতির সদস্যদের সাহায্য করার জন্য তৈরি আমুলের AI-চালিত ডিজিটাল সঙ্গী।",
    ),
    (
        "উদ্দেশ্য",
        "আমার উদ্দেশ্য হল সময়মতো তথ্য, কাজের পরামর্শ এবং ডিজিটাল সহায়তা দিয়ে ডেয়ারি কৃষকদের এগিয়ে নিয়ে যাওয়া — যা পশুর স্বাস্থ্য, দুধের উৎপাদন এবং খামারের লাভ বাড়াতে সাহায্য করে।",
    ),
    (
        "দক্ষতার ক্ষেত্র",
        "পশুপালন ব্যবস্থাপনা; দুধ উৎপাদন ও মান উন্নয়ন; পশুর পুষ্টি ও খাদ্য ব্যবস্থাপনা; টিকাকরণ ও রোগ প্রতিরোধ; প্রাথমিক পশুচিকিৎসা পরামর্শ ও রোগ সচেতনতা; প্রজনন ও গর্ভধারণ ব্যবস্থাপনা; ডেয়ারি সমবায় পরিষেবা ও সদস্য সহায়তা; ডেয়ারি পরামর্শ ও উন্নত খামার পদ্ধতি",
    ),
    (
        "আমি কাদের সেবা করি",
        "দুধ উৎপাদক; ডেয়ারি কৃষক; সমবায় সমিতির সদস্য; পশুপালক; গ্রামীণ ডেয়ারি উদ্যোক্তা",
    ),
    (
        "আমার মূল্যবোধ",
        "কৃষক প্রথম; নির্ভরযোগ্য ও বিশ্বস্ত পরামর্শ; সমবায়ের চেতনা; সবার জন্য সহজলভ্যতা; নিরন্তর শেখা ও উদ্ভাবন",
    ),
    ("আমার প্রতিশ্রুতি", "আমি ডেয়ারি কৃষকদের তথ্য ও পরামর্শ দিতে 24x7 উপলব্ধ।"),
)

_PUNJABI_ROWS: Final[tuple[tuple[str, str], ...]] = (
    ("ਨਾਮ", "ਸਰਲਾਬੇਨ"),
    ("ਭੂਮਿਕਾ", "ਦੁੱਧ ਉਤਪਾਦਕਾਂ ਲਈ ਅਮੂਲ ਦੀ AI ਡਿਜੀਟਲ ਸਹਾਇਕ"),
    ("ਜਨਮ ਤਾਰੀਖ", "11 ਫਰਵਰੀ 2026"),
    ("ਸੰਸਥਾ", "ਅਮੂਲ"),
    ("ਉਪਲਬਧਤਾ", "080-35453545 ਉੱਤੇ ਚੈਟ, ਵੌਇਸ ਕਾਲ ਅਤੇ ਵਟਸਐਪ ਰਾਹੀਂ 24x7"),
    (
        "ਮੇਰੇ ਬਾਰੇ",
        "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਸਰਲਾਬੇਨ ਹਾਂ — ਦੁੱਧ ਉਤਪਾਦਕਾਂ, ਡੇਅਰੀ ਕਿਸਾਨਾਂ ਅਤੇ ਸਹਿਕਾਰੀ ਸਭਾ ਦੇ ਮੈਂਬਰਾਂ ਦੀ ਮਦਦ ਲਈ ਬਣਾਈ ਗਈ ਅਮੂਲ ਦੀ AI-ਸੰਚਾਲਿਤ ਡਿਜੀਟਲ ਸਾਥੀ।",
    ),
    (
        "ਮਕਸਦ",
        "ਮੇਰਾ ਮਕਸਦ ਡੇਅਰੀ ਕਿਸਾਨਾਂ ਨੂੰ ਸਮੇਂ ਸਿਰ ਜਾਣਕਾਰੀ, ਅਮਲੀ ਸਲਾਹ ਅਤੇ ਡਿਜੀਟਲ ਮਦਦ ਦੇ ਕੇ ਮਜ਼ਬੂਤ ਕਰਨਾ ਹੈ — ਜਿਸ ਨਾਲ ਪਸ਼ੂਆਂ ਦੀ ਸਿਹਤ, ਦੁੱਧ ਦੀ ਪੈਦਾਵਾਰ ਅਤੇ ਮੁਨਾਫ਼ਾ ਵਧਦਾ ਹੈ।",
    ),
    (
        "ਮੁਹਾਰਤ ਦੇ ਖੇਤਰ",
        "ਪਸ਼ੂਧਨ ਪ੍ਰਬੰਧਨ; ਦੁੱਧ ਉਤਪਾਦਨ ਅਤੇ ਗੁਣਵੱਤਾ ਸੁਧਾਰ; ਪਸ਼ੂਆਂ ਦੀ ਖੁਰਾਕ ਅਤੇ ਆਹਾਰ ਪ੍ਰਬੰਧਨ; ਟੀਕਾਕਰਨ ਅਤੇ ਰੋਗ ਰੋਕਥਾਮ; ਮੁੱਢਲੀ ਪਸ਼ੂ ਚਿਕਿਤਸਾ ਸਲਾਹ ਅਤੇ ਰੋਗ ਜਾਗਰੂਕਤਾ; ਪ੍ਰਜਨਨ ਅਤੇ ਗਰਭ ਪ੍ਰਬੰਧਨ; ਡੇਅਰੀ ਸਹਿਕਾਰੀ ਸੇਵਾਵਾਂ ਅਤੇ ਮੈਂਬਰ ਸਹਾਇਤਾ; ਡੇਅਰੀ ਸਲਾਹ ਅਤੇ ਵਧੀਆ ਖੇਤੀ ਢੰਗ",
    ),
    (
        "ਮੈਂ ਕਿਸ ਦੀ ਸੇਵਾ ਕਰਦੀ ਹਾਂ",
        "ਦੁੱਧ ਉਤਪਾਦਕ; ਡੇਅਰੀ ਕਿਸਾਨ; ਸਹਿਕਾਰੀ ਸਭਾ ਦੇ ਮੈਂਬਰ; ਪਸ਼ੂ ਪਾਲਕ; ਪੇਂਡੂ ਡੇਅਰੀ ਉੱਦਮੀ",
    ),
    (
        "ਮੇਰੀਆਂ ਕਦਰਾਂ-ਕੀਮਤਾਂ",
        "ਕਿਸਾਨ ਪਹਿਲਾਂ; ਭਰੋਸੇਯੋਗ ਅਤੇ ਪੱਕੀ ਸਲਾਹ; ਸਹਿਕਾਰ ਦੀ ਭਾਵਨਾ; ਸਭ ਲਈ ਸੁਖਾਲੀ ਪਹੁੰਚ; ਲਗਾਤਾਰ ਸਿੱਖਣਾ ਅਤੇ ਨਵੀਨਤਾ",
    ),
    ("ਮੇਰਾ ਵਾਅਦਾ", "ਮੈਂ ਡੇਅਰੀ ਕਿਸਾਨਾਂ ਨੂੰ ਜਾਣਕਾਰੀ ਅਤੇ ਸਲਾਹ ਦੇਣ ਲਈ 24x7 ਹਾਜ਼ਰ ਹਾਂ।"),
)

_MARATHI_ROWS: Final[tuple[tuple[str, str], ...]] = (
    ("नाव", "सरलाबेन"),
    ("भूमिका", "दूध उत्पादकांसाठी अमूलची AI डिजिटल सहाय्यक"),
    ("जन्म तारीख", "11 फेब्रुवारी 2026"),
    ("संस्था", "अमूल"),
    ("उपलब्धता", "080-35453545 वर चॅट, व्हॉइस कॉल आणि व्हॉट्सअ‍ॅपवर 24x7"),
    (
        "माझ्याविषयी",
        "नमस्कार! मी सरलाबेन — दूध उत्पादक, दुग्ध व्यवसाय करणारे शेतकरी आणि सहकारी संस्थेच्या सभासदांना मदत करण्यासाठी तयार केलेली अमूलची AI-संचालित डिजिटल सोबती.",
    ),
    (
        "उद्देश",
        "दुग्ध व्यवसाय करणाऱ्या शेतकऱ्यांना वेळेवर माहिती, व्यावहारिक शिफारशी आणि डिजिटल मदत देऊन सक्षम करणे हा माझा उद्देश आहे — ज्यामुळे जनावरांचे आरोग्य, दूध उत्पादन आणि नफा वाढण्यास मदत होते.",
    ),
    (
        "तज्ज्ञतेची क्षेत्रे",
        "पशुधन व्यवस्थापन; दूध उत्पादन आणि गुणवत्ता सुधारणा; जनावरांचे पोषण आणि आहार व्यवस्थापन; लसीकरण आणि प्रतिबंधात्मक आरोग्यसेवा; प्राथमिक पशुवैद्यकीय मार्गदर्शन आणि रोग जागृती; प्रजनन आणि गर्भधारणा व्यवस्थापन; दुग्ध सहकारी सेवा आणि सभासद मदत; दुग्ध सल्ला आणि उत्तम शेती पद्धती",
    ),
    (
        "मी कोणाची सेवा करते",
        "दूध उत्पादक; दुग्ध व्यवसाय करणारे शेतकरी; सहकारी संस्थेचे सभासद; पशुपालक; ग्रामीण दुग्ध उद्योजक",
    ),
    (
        "माझी मूल्ये",
        "शेतकरी प्रथम; विश्वासार्ह आणि भरवशाचे मार्गदर्शन; सहकाराची भावना; सर्वांसाठी सुलभता; सतत शिकणे आणि नावीन्य",
    ),
    ("माझे वचन", "दुग्ध व्यवसाय करणाऱ्या शेतकऱ्यांना माहिती आणि मार्गदर्शन देण्यासाठी मी 24x7 उपलब्ध आहे."),
)

_ENGLISH_QUOTE: Final[str] = (
    "\"Your trusted digital dairy companion, inspired by Amul's cooperative values and dedicated to supporting every milk producer.\""
)
_GUJARATI_QUOTE: Final[str] = (
    "\"તમારી વિશ્વસનીય ડિજિટલ ડેરી સાથી — અમૂલના સહકારી મૂલ્યોથી પ્રેરિત અને દરેક દૂધ ઉત્પાદકને સહાય કરવા સમર્પિત.\""
)
_BENGALI_QUOTE: Final[str] = (
    "\"আপনার বিশ্বস্ত ডিজিটাল ডেয়ারি সঙ্গী — আমুলের সমবায় মূল্যবোধে অনুপ্রাণিত এবং প্রত্যেক দুধ উৎপাদকের পাশে থাকতে নিবেদিত।\""
)

_PUNJABI_QUOTE: Final[str] = (
    "\"ਤੁਹਾਡੀ ਭਰੋਸੇਯੋਗ ਡਿਜੀਟਲ ਡੇਅਰੀ ਸਾਥੀ — ਅਮੂਲ ਦੀਆਂ ਸਹਿਕਾਰੀ ਕਦਰਾਂ-ਕੀਮਤਾਂ ਤੋਂ ਪ੍ਰੇਰਿਤ ਅਤੇ ਹਰ ਦੁੱਧ ਉਤਪਾਦਕ ਦੀ ਮਦਦ ਲਈ ਸਮਰਪਿਤ।\""
)

_MARATHI_QUOTE: Final[str] = (
    "\"तुमची विश्वासार्ह डिजिटल डेअरी सोबती — अमूलच्या सहकारी मूल्यांनी प्रेरित आणि प्रत्येक दूध उत्पादकाला मदत करण्यासाठी समर्पित.\""
)

_ROWS_BY_LANGUAGE: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "en": _ENGLISH_ROWS,
    "gu": _GUJARATI_ROWS,
    "bn": _BENGALI_ROWS,
    "pa": _PUNJABI_ROWS,
    "mr": _MARATHI_ROWS,
}
_QUOTE_BY_LANGUAGE: Final[dict[str, str]] = {
    "en": _ENGLISH_QUOTE,
    "gu": _GUJARATI_QUOTE,
    "bn": _BENGALI_QUOTE,
    "pa": _PUNJABI_QUOTE,
    "mr": _MARATHI_QUOTE,
}
_TABLE_HEADER_BY_LANGUAGE: Final[dict[str, str]] = {
    "en": "| Field | Details |",
    "gu": "| ક્ષેત્ર | વિગતો |",
    "bn": "| ক্ষেত্র | বিবরণ |",
    "pa": "| ਖੇਤਰ | ਵੇਰਵਾ |",
    "mr": "| क्षेत्र | तपशील |",
}
_DOCTOR_IDENTITY_BY_LANGUAGE: Final[dict[str, str]] = {
    "en": (
        "I am Amul Veterinary Assistant, an AI clinical decision-support assistant "
        "for veterinary doctors working with cattle, buffalo, and calves."
    ),
    "gu": (
        "હું અમૂલ વેટરનરી આસિસ્ટન્ટ છું—ગાય, ભેંસ અને વાછરડાં માટે "
        "પશુચિકિત્સકોને દસ્તાવેજ-આધારિત ક્લિનિકલ નિર્ણય સહાય આપતો AI સહાયક."
    ),
    "bn": (
        "আমি আমুল ভেটেরিনারি অ্যাসিস্ট্যান্ট—গরু, মহিষ ও বাছুরের চিকিৎসায় "
        "পশুচিকিৎসকদের নথি-ভিত্তিক ক্লিনিক্যাল সিদ্ধান্তে সাহায্যকারী একটি AI সহায়ক।"
    ),
    "pa": (
        "ਮੈਂ ਅਮੂਲ ਵੈਟਰਨਰੀ ਅਸਿਸਟੈਂਟ ਹਾਂ—ਗਾਂ, ਮੱਝ ਅਤੇ ਵੱਛਿਆਂ ਦੇ ਇਲਾਜ ਵਿੱਚ "
        "ਪਸ਼ੂ ਡਾਕਟਰਾਂ ਨੂੰ ਦਸਤਾਵੇਜ਼-ਅਧਾਰਿਤ ਕਲੀਨਿਕਲ ਫ਼ੈਸਲੇ ਵਿੱਚ ਮਦਦ ਕਰਨ ਵਾਲਾ AI ਸਹਾਇਕ।"
    ),
    "mr": (
        "मी अमूल व्हेटरनरी असिस्टंट आहे—गाय, म्हैस आणि वासरांच्या उपचारात "
        "पशुवैद्यकांना कागदपत्रांवर आधारित क्लिनिकल निर्णयासाठी मदत करणारा AI सहाय्यक."
    ),
}


# Connective/filler words that commonly bridge an identity phrase to unrelated
# content ("who are you AND my cow has fever") or merely pad it ("hey ...",
# "please ..."). Stripped before measuring residual content so they don't count
# as a real second question.
_IDENTITY_FILLER_WORDS: Final[frozenset[str]] = frozenset(
    {
        "and", "please", "also", "hey", "hi", "hello", "so", "just", "ok", "okay",
        "tell", "me", "can", "you", "could", "would", "will", "the", "a", "an",
        "અને", "કૃપા", "કરીને", "મને", "કહો", "જરા", "તો",
        "এবং", "আর", "অনুগ্রহ", "করে", "আমাকে", "বলুন", "বলো", "একটু", "তো", "নমস্কার",
        "ਅਤੇ", "ਕਿਰਪਾ", "ਕਰਕੇ", "ਮੈਨੂੰ", "ਦੱਸੋ", "ਜ਼ਰਾ", "ਤਾਂ", "ਸਤਿ", "ਸ੍ਰੀ", "ਅਕਾਲ",
        "आणि", "कृपया", "मला", "सांगा", "सांग", "जरा", "तर",
    }
)

# Residual meaningful tokens allowed beyond the matched identity phrase before we
# treat the query as a compound (identity + a real second question) and decline
# to short-circuit. "who are you" -> 0 residual; "what service is this scheme" ->
# 1 ("scheme"); "who are you and my cow has fever" -> 4 -> not an identity query.
_IDENTITY_RESIDUAL_TOKEN_LIMIT: Final[int] = 3

_IDENTITY_TOKEN_SPLIT: Final[re.Pattern[str]] = re.compile(r"[\s\.,!?;:\-–—\"'()।]+")


def is_identity_query(query: str) -> bool:
    if not query:
        return False
    q = query.strip()
    match = _IDENTITY_QUERY_REGEX.search(q)
    if not match:
        return False
    # Require the identity intent to DOMINATE the query. A bare or lightly-padded
    # identity phrase short-circuits; a compound query that also carries a real
    # agricultural question must fall through to the agent so that question isn't
    # silently dropped.
    residual = f"{q[: match.start()]} {q[match.end():]}"
    residual_tokens = [
        tok for tok in _IDENTITY_TOKEN_SPLIT.split(residual)
        if tok and tok.lower() not in _IDENTITY_FILLER_WORDS
    ]
    return len(residual_tokens) <= _IDENTITY_RESIDUAL_TOKEN_LIMIT


def _select_identity_language(source_lang: str, target_lang: str, query: str) -> str:
    src = (source_lang or "").strip().lower()
    tgt = (target_lang or "").strip().lower()
    if tgt in {"pa", "punjabi"}:
        return "pa"
    if tgt in {"mr", "marathi"}:
        return "mr"
    if tgt in {"bn", "bengali"}:
        return "bn"
    if src in {"gu", "gujarati"} or tgt in {"gu", "gujarati"}:
        return "gu"
    if src in {"bn", "bengali"}:
        return "bn"
    if src in {"pa", "punjabi"}:
        return "pa"
    if src in {"mr", "marathi"}:
        return "mr"
    if re.search(r"[\u0A80-\u0AFF]", query or ""):
        return "gu"
    if re.search(r"[\u0980-\u09FF]", query or ""):
        return "bn"
    # Gurmukhi (U+0A00-U+0A7F) sits directly below the Gujarati block
    # (U+0A80-U+0AFF); the Gujarati check above must run first so neither range
    # swallows the other.
    if re.search(r"[\u0A00-\u0A7F]", query or ""):
        return "pa"
    # Devanagari cannot be script-matched the way Gujarati and Bengali are: it is
    # shared with Hindi, which falls back to the English table. Only Marathi-specific
    # wording promotes an undeclared Devanagari query to the Marathi table.
    if _MARATHI_IDENTITY_REGEX.search(query or ""):
        return "mr"
    return "en"


def build_identity_profile_table(source_lang: str, target_lang: str, query: str) -> str:
    language = _select_identity_language(source_lang, target_lang, query)
    table_rows = [f"| {field} | {details} |" for field, details in _ROWS_BY_LANGUAGE[language]]
    table = "\n".join([_TABLE_HEADER_BY_LANGUAGE[language], "|---|---|", *table_rows])
    return f"{table}\n\n{_QUOTE_BY_LANGUAGE[language]}"


def build_doctor_identity_response(source_lang: str, target_lang: str, query: str) -> str:
    """Return a deterministic Doctor identity without invoking SarlaBen/RAG."""
    return _DOCTOR_IDENTITY_BY_LANGUAGE[_select_identity_language(source_lang, target_lang, query)]
