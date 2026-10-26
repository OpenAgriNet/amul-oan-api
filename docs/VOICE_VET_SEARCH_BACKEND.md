# Veterinary retrieval in voice

Set `VET_SEARCH_BACKEND=beckn` in the deployment to route the registered voice
`search_documents` tool through the same veterinary Beckn provider as chat.
`AMUL_NETWORK_URL` must resolve to the seeker from the app container. The provider
selects Marqo or Qdrant and applies its existing ranking and diversity policy.
Query validation and the tool's result limit remain in place.

The default is `marqo`, preserving existing deployments. Unknown values fail
through the existing tool retry contract; a failed Beckn request does not silently
switch back to Marqo. Switching this read path does not configure pipeline writes.

DEV voice uses its configured public JWT verification key. Obtain a test token
from the matching chat issuer rather than mounting its signing key into voice.
Voice's anonymous-token endpoint cannot issue tokens in a verification-only
deployment.
