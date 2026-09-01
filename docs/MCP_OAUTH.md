# ChatGPT OAuth setup for Order Extractor

The backend now implements the OAuth **resource-server** side of the connection.
Auth0 handles login, consent, authorization codes, PKCE and token issuance.
ChatGPT handles account linking and presents access tokens to /mcp.

Your tenant's public discovery endpoint was checked successfully:

- Issuer: https://dev-y6rh0x822nwefwin.us.auth0.com/
- Resource/API identifier: https://order-extractor-kdih.onrender.com/mcp
- Auth0 application: **Order Extractor ChatGPT**, Regular Web Application.

The application and API are different Auth0 records: the application represents
ChatGPT as an OAuth client; the API represents the protected MCP endpoint.
This setup uses a predefined OAuth client. Dynamic client registration is not
required, and no new authorization server is implemented in this repository.

## 1. Configure the Auth0 API

In Applications > APIs, create **Order Extractor MCP** with the exact resource
identifier above and RS256 signing. In its Permissions tab, add:

Use **Add a Permission** at the top (Permission + Description, then Add).
The entries must appear under **List of Permissions**. The lower
**Authorization Details Types** section is a different feature and is not used
by this integration.

| Permission | Description |
|---|---|
| orders:read | Read factory orders, processing jobs, summaries and artifacts |
| orders:write | Create drafts and perform permitted, confirmed workflow actions |

In the API Settings, enable **RBAC** and **Add Permissions in the Access Token**.
If selecting a token dialect, use the RFC 9068 profile with authorization
permissions (rfc9068_profile_authz). Both the legacy Auth0 JWT and at+jwt header
types are accepted, provided all required claims are present.

Set both API token expiration settings to **900 seconds**. The backend rejects
tokens whose declared lifetime exceeds 3,600 seconds. Enable offline access
only if refresh tokens are desired; the client must also request that scope.

In **Application Access Policy**, keep **User-delegated Access** set to
**Per-app authorization**. Save the settings. In the API's **Application Access**
tab, find **Order Extractor ChatGPT**, select **Edit**, and **Grant Access** under
**User-Delegated Access**. Select only orders:read and orders:write, then save.
Leave the application's Client Access ungranted and the default permissions for
third-party applications Unauthorized. This predefined client signs users in;
it does not need machine-to-machine access. See
[Auth0 API access policies](https://auth0.com/docs/get-started/apis/api-access-policies-for-applications).

See [Auth0's MCP authorization guide](https://auth0.com/ai/docs/mcp/get-started/authorization-for-your-mcp-server).

## 2. Configure tenant and application settings

In tenant Settings > Advanced, enable:

- **Resource Parameter Compatibility Profile**, so ChatGPT's resource parameter
  selects the exact API audience.
- **Include Issuer in Authorization Responses**, so authorization responses identify
  the issuing tenant.

In the Order Extractor ChatGPT application:

- Enable the chosen login connection in Connections.
- Use the authorization-code grant; enable refresh-token grant if needed.
- Keep the Client Secret private. It is entered directly into ChatGPT's OAuth
  connection settings when linking, not into this repository, a tool prompt,
  an MCP response, or the Render backend.
- Copy the exact Client ID from Settings into the backend client allowlist later.
- Leave callback setup until ChatGPT shows the exact callback for its connection.
  Then add that exact value to Allowed Callback URLs. Do not guess a callback ID
  or use a wildcard.

Current ChatGPT may show a stable callback when the issuer supports response
identification, or a connection-specific callback otherwise. Follow the value
shown in ChatGPT. Details are in the [OpenAI authentication guide](https://developers.openai.com/plugins/build/auth).

## 3. Authorize the factory users

The Auth0 dashboard administrator account does not automatically become an
application login account.

1. Under User Management > Users, create the intended factory login user, or let
   that user sign in through an enabled Auth0 connection.
2. Create a role named **Order Extractor Operator**. Add both API permissions.
   A viewer role should receive only orders:read.
3. Assign the relevant role to each approved user.
4. Copy each user's exact **user_id**, for example auth0|... or google-oauth2|...,
   into ORDER_EXTRACTOR_MCP_ALLOWED_SUBJECTS. This is an identity allowlist, not
   a password. Do not substitute the user's email address or the application ID.

An allowed user also needs the appropriate role permissions AND granted OAuth
scopes. The backend intersects the scope and permissions claims. A token merely
requesting orders:write does not give a viewer write permission.

## 4. Backend configuration for the later deployment

Set these in Render's environment after the code is reviewed and deployment
is authorized. Replace the two placeholders with copied Auth0 values.

Use **Save only** while preparing these variables; deploy after the build/root
settings in [MCP_INTEGRATION.md](MCP_INTEGRATION.md) are ready.
For the verified Render service, also retain `DB_DIR=/var/data`, set
`PYTHON_VERSION=3.13.4` and `NODE_VERSION=22`, and set
`ORDER_EXTRACTOR_MCP_DURABLE_STORAGE=true` for the confirmed persistent disk.

    ORDER_EXTRACTOR_MCP_ENABLED=true
    ORDER_EXTRACTOR_MCP_AUTH_MODE=oauth
    ORDER_EXTRACTOR_MCP_OAUTH_ISSUER=https://dev-y6rh0x822nwefwin.us.auth0.com/
    ORDER_EXTRACTOR_MCP_RESOURCE_URL=https://order-extractor-kdih.onrender.com/mcp
    ORDER_EXTRACTOR_MCP_ALLOWED_SUBJECTS=<approved Auth0 user_id>
    ORDER_EXTRACTOR_MCP_OAUTH_CLIENT_IDS=<Order Extractor ChatGPT Client ID>
    ORDER_EXTRACTOR_MCP_READ_ONLY=true

Comma-separate additional approved user IDs or OAuth client IDs. Keep read-only
mode enabled for the first connection check, then deliberately enable writes.

No Auth0 Client Secret or Management API credential is required by the backend:
it verifies signatures with Auth0's public signing keys. The private bearer token
is ignored in OAuth mode. Authentication modes are exclusive, with no fallback.

The public discovery routes are:

- /.well-known/oauth-protected-resource/mcp
- /.well-known/oauth-protected-resource (compatibility alias)

Both return only public OAuth metadata. Every /mcp request and artifact download
still requires a valid access token. Discovery does not expose order data.

## 5. Run the readiness check

From the repository root, after supplying the environment configuration:

    .venv/bin/python backend/mcp_oauth_check.py --offline
    .venv/bin/python backend/mcp_oauth_check.py

The first command validates configuration without network access. The second
checks public issuer metadata, PKCE support, endpoints, client-registration
options and signing keys. It never requests a user token or reads a client secret.
The result includes the provider settings that still require manual verification;
public metadata cannot prove that user roles or the API audience are configured.

Use the local installation instructions in MCP_INTEGRATION.md. For a real local
OAuth test, set the resource URL to the stable HTTPS tunnel's /mcp URL and create
a matching development API identifier in Auth0. A local HTTP listener can sit
behind that HTTPS tunnel, but the advertised resource must use HTTPS. Do not
reuse the production API identifier for a different endpoint.

## 6. Connect ChatGPT after deployment

1. Enable Settings > Security and login > Developer mode in ChatGPT.
2. Open Plugins, select +, and create an Order Extractor connection.
3. Enter the deployed HTTPS /mcp address and choose OAuth.
4. Supply the predefined Auth0 Client ID and Client Secret directly in ChatGPT's
   OAuth configuration. Keep the secret out of conversation messages.
5. Copy ChatGPT's displayed redirect URI into the Auth0 application's Allowed
   Callback URLs and save it.
6. Complete linking by signing in as the approved factory user and granting access.
7. Confirm 18 tools are visible. Start with platform_health and list_manual_orders.
8. With writes enabled on the backend, use a designated test draft to verify the
   separate create, approve and processing actions. Existing status/version and
   explicit-confirmation rules continue to apply.

ChatGPT may request read access first and prompt again for write permission.
Tool descriptions advertise their scopes, and denied calls return the
mcp/www_authenticate metadata needed to request additional permission.

After changing auth policy or tool metadata, refresh the connection and start
a new conversation. The exact UI may vary with account/workspace availability.
See [ChatGPT Developer Mode](https://developers.openai.com/api/docs/guides/developer-mode).

## Security behavior and operational limits

- RS256 signatures from configured Auth0 public keys only; issuer, resource
  audience, expiry, issued-at and not-before claims are checked.
- User IDs are explicitly allowed; machine/service-account tokens are rejected.
- Client ID restrictions are enforced when configured. Leaving the client list
  empty deliberately permits other OAuth clients for the approved users.
- Read tools and artifact downloads require orders:read. Every write tool requires
  orders:read and orders:write, plus the existing business safeguards.
- Token lifetime is at most one hour; clock skew allowance is 30 seconds.
- Public keys are cached for five minutes. Unknown key IDs can trigger at most
  one refresh every ten seconds. Expired caches fail closed during provider outages.
- Public-key fetches have a five-second overall deadline, a 256 KB response cap,
  TLS verification, no redirects and no inherited proxy credentials.
- Audit actors use a stable hash of issuer + a null separator + user ID; tokens,
  emails and raw user IDs are not written into MCP audit rows. Idempotency keys
  are isolated per resolved actor, across token refreshes.
- Removing a user from the backend allowlist and restarting blocks that user
  immediately. Changing Auth0 roles may not affect an already issued token until
  it expires or refreshes. There is no token introspection call on every request.
- Changes to issuer, audience, users, clients or authentication mode need a restart.
  The existing read-only switch and workflow limits still apply.

This authenticates the MCP surface. It does not retrofit authentication onto
historical REST endpoints. Deployment must retain the existing ingress decisions.

## Troubleshooting

| Symptom | Check |
|---|---|
| 503 MCP_UNAVAILABLE | Enabled flag, mode, HTTPS issuer/resource, nonempty explicit user allowlist |
| 503 AUTH_PROVIDER_UNAVAILABLE | Outbound HTTPS to the configured Auth0 JWKS; retry after ten seconds |
| 401 INVALID_TOKEN | RS256 API access token, exact issuer/audience, timestamps, permissions claim, lifetime <= 3600s |
| 403 FORBIDDEN | Exact user_id, OAuth client allowlist, or backend read-only mode |
| INSUFFICIENT_SCOPE | Requested scopes and assigned role permissions must both contain the required values |
| Callback mismatch | Copy the exact current URI from ChatGPT into Auth0; do not infer it |
| Opaque token/wrong audience | Enable Resource Parameter Compatibility Profile and use the exact API identifier |
| Repeated permission prompt | Enable API RBAC + permissions claim, assign roles, then re-link for a fresh token |
| App unauthorized or empty scope list | Define actual API Permissions, then grant the predefined application User-Delegated Access to both scopes |

## Validation scope

The automated suite uses real RSA signatures and the official MCP Python OAuth
client with a local simulated issuer to exercise discovery, PKCE, token exchange,
scope enforcement and a real MCP tool call. It does not claim that the Auth0
dashboard or ChatGPT production linking has been completed.

The user's real tenant's public OIDC discovery was read successfully, and the
application/API, both API permissions, the predefined application's user grant,
the operator role assignment, login connection and tenant compatibility switches
were confirmed in dashboard screenshots. Both token lifetime fields were set to
900 seconds in a screenshot that still showed an unsaved-changes notice; save
that page. A later public provider readiness check passed, including signing keys,
S256 and issuer identification. The exact ChatGPT callback and a real end-user
token still need verification during the first hosted connection.
No GitHub push or Render deployment was performed
as part of this OAuth implementation.
