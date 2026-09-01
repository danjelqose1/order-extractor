"""Read-only Auth0 readiness check; never requests user access tokens."""
from __future__ import annotations

import argparse
import asyncio
import json
import jwt
from mcp_auth import OAuthSettings, _https_url, fetch_public_json


async def check_provider(settings):
    metadata = await fetch_public_json(settings.issuer + ".well-known/openid-configuration")
    checks = {
        "issuer_matches": metadata.get("issuer") == settings.issuer,
        "pkce_s256": "S256" in metadata.get("code_challenge_methods_supported", []),
        "jwks_matches": metadata.get("jwks_uri") == settings.jwks_url,
        "code_flow": "code" in metadata.get("response_types_supported", []),
        "client_registration": bool(settings.allowed_clients or metadata.get("client_id_metadata_document_supported")
                                    or metadata.get("registration_endpoint")),
    }
    for field in ("authorization_endpoint", "token_endpoint"):
        try:
            url = _https_url(metadata.get(field, ""), field)
            checks[field] = url.netloc == _https_url(settings.issuer, "issuer").netloc
        except (ValueError, TypeError):
            checks[field] = False
    keys = await fetch_public_json(settings.jwks_url)
    checks["rsa_signing_key"] = bool(any(
        item.get("kty") == "RSA" and item.get("alg", "RS256") == "RS256" and item.get("kid")
        and jwt.PyJWK.from_dict(item, algorithm="RS256").key.key_size >= 2048
        for item in keys.get("keys", []) if isinstance(item, dict)
    ))
    return dict(ok=all(checks.values()), checks=checks,
                issuer_identification=metadata.get("authorization_response_iss_parameter_supported") is True,
                manual_checks=["Resource Parameter Compatibility Profile enabled",
                               "API audience equals the exact MCP resource URL",
                               "RBAC and permissions claim enabled; user roles assigned",
                               "Token lifetime at most 3600 seconds",
                               "Exact ChatGPT callback copied from its connection settings"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Validate configuration without contacting Auth0")
    args = parser.parse_args()
    try:
        settings = OAuthSettings.from_env()
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    if args.offline:
        print(json.dumps({"ok": True, "approved_user_count": len(settings.allowed_subjects),
                          "resource_metadata": settings.metadata()}))
        return 0
    try:
        result = asyncio.run(check_provider(settings))
    except Exception:
        print(json.dumps({"ok": False, "error": "Provider discovery or signing-key check failed; verify the issuer and network connection."}))
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
