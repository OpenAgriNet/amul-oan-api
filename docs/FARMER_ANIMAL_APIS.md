# Farmer and Animal APIs

The farmer and animal tools use **two backends** and return a single cohesive response. At least one of `PASHUGPT_TOKEN` or `PASHUGPT_TOKEN_3` must be set in `.env`.

## Backends

| Backend        | Env var           | Farmer endpoint                         | Animal endpoint                          |
|----------------|-------------------|-----------------------------------------|-----------------------------------------|
| amulpashudhan  | `PASHUGPT_TOKEN`  | GetFarmerDetailsByMobile?mobileNumber=   | GetAnimalDetailsByTagNo?tagNo=           |
| herdman.live   | `PASHUGPT_TOKEN_3` | get-amul-farmer?mobileno=              | get-amul-animal?TagID=                   |

- **Farmer**: Tries amulpashudhan first; if 204/empty/error, tries herdman. Results are merged and deduplicated by `(societyName, farmerCode)`.
- **Animal**: Tries amulpashudhan first; if 204/empty/error, tries herdman. When both return data, fields are merged (amulpashudhan preferred; herdman fills missing).

## Behaviour

- **Phone numbers**: Normalized to digits only (e.g. `+91 94157 87824` → `9415787824`).
- **Tag numbers**: Trimmed of whitespace.
- **204 No Content / empty body**: Treated as “no data”; fallback backend is tried.
- **Non-200 / JSON errors**: Logged; fallback backend is tried. User gets a clear “no data” message only if both fail or return nothing.
- **Mutually redundant info**: Farmer records deduplicated; animal response is one merged object.

## Farmer response shape (amulpashudhan)

Array of records with:

- `state`, `district`, `subDistrict`, `village`, `unionName`, `societyName`
- `farmerName`, `mobileNumber`, `farmerCode`
- `avgMilkPerDayCow`, `avgMilkPerDayBuff`, `cowSnf`, `cowFat`, `buffSnf`, `buffFat`
- `tagNo` (comma-separated or null), `totalAnimals`, `cow`, `buffalo`, `totalMilkingAnimals`

Some fields can be null (e.g. state/district/village).

## Animal response shapes

**amulpashudhan** (single object):

- `tagNumber`, `animalType`, `breed`, `milkingStage`, `pregnancyStage`, `dateOfBirth`
- `lactationNo`, `lastBreedingActivity`, `lastHealthActivity`

**herdman** (wrapped in `Animal` array; mapped to canonical keys):

- `tagno` → `tagNumber`, `Animal Type` → `animalType`, `Breed` → `breed`
- `Milking Stage`, `DOB` → `dateOfBirth`, `Currant Lactation no` → `lactationNo`
- `Last AI`, `Last PD`, `Last Calvingdate`, `Farmer complaint`, `Diagnosis`, `Medicine Given`

Merged output uses the canonical keys above; herdman-only fields appear when present.

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

## Exploration

To capture raw responses from both APIs (for new phone/tag sets):

```bash
python scripts/explore_farmer_animal_apis.py
```

Outputs:

- `exploration/farmer_responses_<timestamp>.json`
- `exploration/animal_responses_<timestamp>.json`

Edit `PHONE_NUMBERS` and `TAG_NUMBERS` in the script to probe different values.

## Env vars (summary)

- `PASHUGPT_TOKEN`: amulpashudhan (farmer + animal).
- `PASHUGPT_TOKEN_2`: CVCC health API (see `get_cvcc_health_details`); not used by farmer/animal tools.
- `PASHUGPT_TOKEN_3`: herdman.live (farmer + animal).

At least one of `PASHUGPT_TOKEN` or `PASHUGPT_TOKEN_3` must be set for farmer and animal tools.

## Exploration summary (Jan 2026)

- **Farmer**: For phones 9415787824, 9375028676, 9035395028 only amulpashudhan returned data; herdman returned `[]`. So farmer tool effectively uses amulpashudhan when both tokens are set; herdman is tried on failure/empty.
- **Animal**: amulpashudhan returns 200 for tags like 105183817302, 107304847832 and 204 for others; herdman returns `[]` for most tags but returns data for **340122347792** (different schema: Animal array, Last AI/PD/Calvingdate, Diagnosis, Medicine Given). So tag coverage differs by backend; the tool merges when both return data.
