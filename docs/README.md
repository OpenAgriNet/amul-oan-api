# Documentation Index

This folder contains the maintained, repo-specific documentation set.

## Current Docs

- `FARMER_ANIMAL_APIS.md`: backend integration notes for farmer and animal data sources.
- `HEAT_ALERT_WEBHOOK.md`: partner heat-alert webhook contract, WhatsApp send, retention and deploy checklist.
- `TRANSLATION_PIPELINE_API.md`: request/response behavior for the translation pipeline.
- `WEBVIEW_URL_API.md`: integration guide for the webview URL endpoint.
- `REPOSITORY_REVIEW_2026-03-10.md`: codebase review notes, cleanup findings, and follow-up recommendations.

Telemetry: this repo sends it. What a chat or voice turn sends is in
`telemetry/contracts/`, and the tests check the code against it. Reading it
(eras, mappings, adapters, the import and the query API), and the rules and
steps for changing what is sent (`docs/STANDARD.md`, `docs/CHANGES.md`), are in
OpenAgriNet/amul-telemetry.

## Documentation Policy

- Keep docs repository-specific and current.
- Do not store credentials, admin passwords, tokens, or internal host access details here.
- Remove review scratchpads and duplicated analysis once the important conclusions are captured in stable docs.
