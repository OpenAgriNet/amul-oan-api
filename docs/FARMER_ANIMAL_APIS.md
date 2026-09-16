# Farmer and Animal APIs

The farmer and animal tools use the **amulpashudhan backend** and return a single cohesive response. `PASHUGPT_TOKEN` must be set in `.env`.

## Backends

| Backend        | Env var           | Farmer endpoint                         | Animal endpoint                          |
|----------------|-------------------|-----------------------------------------|-----------------------------------------|
| amulpashudhan  | `PASHUGPT_TOKEN`  | GetFarmerDetailsByMobile?mobileNumber=   | GetAnimalDetailsByTagNo?tagNo=           |

- **Farmer**: Records are deduplicated by `(societyName, farmerCode)`.
- **Animal**: A single record is returned for the tag.

## Behaviour

- **Phone numbers**: Normalized to digits only (e.g. `+91 94157 87824` → `9415787824`).
- **Tag numbers**: Trimmed of whitespace.
- **204 No Content / empty body**: Treated as “no data”.
- **Non-200 / JSON errors**: Logged; the user gets a clear “no data” message.
- **Mutually redundant info**: Farmer records deduplicated; animal response is one object.

## Farmer response shape (amulpashudhan)

Array of records with:

- `state`, `district`, `subDistrict`, `village`, `unionName`, `societyName`
- `farmerName`, `mobileNumber`, `farmerCode`
- `avgMilkPerDayCow`, `avgMilkPerDayBuff`, `cowSnf`, `cowFat`, `buffSnf`, `buffFat`
- `tagNo` (comma-separated or null), `totalAnimals`, `cow`, `buffalo`, `totalMilkingAnimals`

Some fields can be null (e.g. state/district/village).

## Animal response shape

**amulpashudhan** (single object):

- `tagNumber`, `animalType`, `breed`, `milkingStage`, `pregnancyStage`, `dateOfBirth`
- `lactationNo`, `lastBreedingActivity`, `lastHealthActivity`

## Banas Operated Visit API (GetOperatedVisit)

A **separate backend** (Banas mobile API) returns **completed veterinary visit records** for a given animal tag. Used for visit history, not for farmer/animal master data.

| Backend              | Base URL                              | Endpoint                                      |
|----------------------|----------------------------------------|-----------------------------------------------|
| banasmobileapi.amnex | `https://banasmobileapi.amnex.com`     | `POST /api/FarmerVisitAPIKOS/GetOperatedVisit` |

**Request**

- **Method:** `POST`
- **Content-Type:** `application/json`
- **Body:**
  - `strApiKey` (string): API key (e.g. from Postman collection).
  - `tagId` (string): Animal tag number (e.g. `"340129866427"`).

**Response**

JSON **array** of visit objects. Each object includes:

- **Visit:** `VisitCode`, `VisitNoteDate`, `VisitScheduleDate`, `VisitAllocationDate`, `EntryTime`, `VisitResponseTime` (minutes), `VisitStatus` (e.g. `"Completed"`).
- **Animal:** `animaltagnumber`, `speciesname`, `gendername`, `pregnancystatus`, `breed`, `milkstatus`, `agegroup`, `Ailment1`–`Ailment3`.
- **Member/farmer:** `membername`, `membercode`, `memberaddress`, `membercontactno`, `societyname`, `societycode`, `societyphonenumber`.
- **Clinical:** `disease`, `DiseaseName`, `DiseaseCode1`, `diseasegroup`, `prognosisdetails`, `medicineremarks`.
- **Staff:** `primarydoctorname`, `doctorcode`, `doctorMobile`, `Drivername`, `vehicleregno`.
- **Place:** `VetcentreName`, `AllotedVetCentreName`, `societyaddress`.
- **Payment:** `PaymentOption`, `PaymentMode`, `PaymentComment`.
- **Nested JSON strings:** `MedicinesJson` (array of medicine objects: stock, uomdoctor, uommedicine, medicinename, remarks), `LabReportsJson` (array of lab sample objects).

**Example request**

```bash
curl -X POST "https://banasmobileapi.amnex.com/api/FarmerVisitAPIKOS/GetOperatedVisit" \
  -H "Content-Type: application/json" \
  -d '{"strApiKey": "<API_KEY>", "tagId": "340129866427"}'
```

**Spec / Postman**

- Collection file: `Banas_Operated_Visit_New.postman_collection.json` (root of repo). Import into Postman to run the request; the collection contains the endpoint and example body.

## GetFarmerBonusAmount (amulpashudhan)

Fetches bonus amount(s) credited to a farmer for one or more periods. Used by the chat agent tool `get_farmer_bonus_amount` (signed-in only; codes come from authenticated mobile → farmer accounts, never from the LLM).

| Backend       | Env var          | Endpoint |
|---------------|------------------|----------|
| amulpashudhan | `PASHUGPT_TOKEN` | `GET {AMULPASHUDHAN_BASE_URL}/GetFarmerBonusAmount` |

**Request (query params)**

- `unionCode` (string, required) — union / organization code (alphanumeric)
- `societyCode` (string, required)
- `farmerCode` (string, required)
- Auth: `Authorization: Bearer {PASHUGPT_TOKEN}`, `Accept: application/json`

**Success response (200)**

A **plain JSON array** (not wrapped in `APIStatusCode` / `Message` / `Data`). Each item may include:

- `societyCode`, `societyName`, `societyNameLocal`
- `farmerCode`, `farmerName`, `farmerLocalName`
- `bonusAmount` (decimal)
- `fromDate`, `toDate` (ISO datetime strings)

Empty array `[]` means no bonus records for that account.

**Caveats**

- Currently returns data only for unions whose master data source is **AMCS**. Unions sourced from Akashganga are not supported yet (business error: bonus amount not supported for that union’s data source).
- There is **no Beckn / network path** for this endpoint; the tool calls PashuGPT directly.
- The agent fans out one request per authenticated account and merges successful results.

**Example**

```bash
curl --location \
  "${AMULPASHUDHAN_BASE_URL}/GetFarmerBonusAmount?unionCode=0001&societyCode=2004&farmerCode=0001" \
  --header 'accept: application/json' \
  --header "Authorization: Bearer ${PASHUGPT_TOKEN}"
```

## Exploration

To capture raw responses from the farmer and animal APIs (for new phone/tag sets):

```bash
python scripts/explore_farmer_animal_apis.py
```

Outputs:

- `exploration/farmer_responses_<timestamp>.json`
- `exploration/animal_responses_<timestamp>.json`

Edit `PHONE_NUMBERS` and `TAG_NUMBERS` in the script to probe different values.

## Env vars (summary)

- `PASHUGPT_TOKEN`: amulpashudhan (farmer + animal + milk collection + bonus amount).
- `PASHUGPT_TOKEN_2`: CVCC health API (see `get_cvcc_health_details`); not used by farmer/animal tools.

`PASHUGPT_TOKEN` must be set for the farmer and animal tools, including bonus amount.

## Exploration summary (Jan 2026)

- **Farmer**: Phones 9415787824, 9375028676 and 9035395028 all returned data from amulpashudhan.
- **Animal**: amulpashudhan returns 200 for tags like 105183817302, 107304847832 and 204 for others.
