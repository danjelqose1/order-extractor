"""Private MCP authentication: explicit bearer OR Auth0 OAuth resource-server mode.

Login, PKCE, consent and token issuance belong to the managed authorization server.
No OAuth client secret, password, or token is stored by this resource server.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
import time
from urllib.parse import urlsplit

import httpx
import jwt

READ_SCOPE = "orders:read"
WRITE_SCOPE = "orders:write"
SCOPES = (READ_SCOPE, WRITE_SCOPE)
METADATA_PATH = "/.well-known/oauth-protected-resource/mcp"


class AuthenticationError(Exception):
    def __init__(self, code, message, status=401, scopes=()):
        super().__init__(message)
        self.code, self.message, self.status, self.scopes = code, message, status, scopes


@dataclass(frozen=True)
class Principal:
    actor: str
    scopes: frozenset[str]
    mode: str


def _https_url(value, field):
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or any(c.isspace() or ord(c) < 32 for c in value)
            or any(c in value for c in ('"', "\\", "<", ">"))):
        raise ValueError(f"{field} must be a canonical HTTPS URL without credentials, query or fragment")
    # Force validation of an optional port without exposing the supplied value.
    try:
        parsed.port
    except ValueError:
        raise ValueError(f"{field} has an invalid port") from None
    return parsed


@dataclass(frozen=True)
class OAuthSettings:
    issuer: str
    resource: str
    allowed_subjects: frozenset[str]
    allowed_clients: frozenset[str]

    @classmethod
    def from_env(cls):
        issuer = os.getenv("ORDER_EXTRACTOR_MCP_OAUTH_ISSUER", "").strip()
        resource = os.getenv("ORDER_EXTRACTOR_MCP_RESOURCE_URL", "").strip()
        issuer_url = _https_url(issuer, "ORDER_EXTRACTOR_MCP_OAUTH_ISSUER")
        resource_url = _https_url(resource, "ORDER_EXTRACTOR_MCP_RESOURCE_URL")
        if issuer_url.path != "/":
            raise ValueError("ORDER_EXTRACTOR_MCP_OAUTH_ISSUER must use the Auth0 origin with its trailing slash")
        if resource_url.path != "/mcp":
            raise ValueError("ORDER_EXTRACTOR_MCP_RESOURCE_URL must end with /mcp, without a trailing slash")
        subjects = frozenset(s.strip() for s in os.getenv("ORDER_EXTRACTOR_MCP_ALLOWED_SUBJECTS", "").split(",") if s.strip())
        clients = frozenset(s.strip() for s in os.getenv("ORDER_EXTRACTOR_MCP_OAUTH_CLIENT_IDS", "").split(",") if s.strip())
        if not subjects or "*" in subjects:
            raise ValueError("ORDER_EXTRACTOR_MCP_ALLOWED_SUBJECTS must contain explicit approved Auth0 user IDs")
        if len(subjects) > 100 or any(len(s) > 255 or any(ord(c) < 32 for c in s) for s in subjects):
            raise ValueError("Invalid approved-user list")
        if "*" in clients:
            raise ValueError("OAuth client restrictions cannot contain a wildcard")
        return cls(issuer, resource, subjects, clients)

    @property
    def jwks_url(self):
        return self.issuer + ".well-known/jwks.json"

    @property
    def metadata_url(self):
        url = urlsplit(self.resource)
        return f"{url.scheme}://{url.netloc}{METADATA_PATH}"

    def metadata(self):
        return dict(resource=self.resource, resource_name="Order Extractor",
                    authorization_servers=[self.issuer], scopes_supported=list(SCOPES),
                    bearer_methods_supported=["header"])

    def challenge(self, *, error=None, scopes=(READ_SCOPE,)):
        value = f'Bearer resource_metadata="{self.metadata_url}", scope="{" ".join(scopes)}"'
        if error:
            description = "Additional permission is required." if error == "insufficient_scope" else "Sign in again to continue."
            value += f', error="{error}", error_description="{description}"'
        return value


async def fetch_public_json(url):
    """Bounded public-key/metadata fetch: no credentials, redirects or proxy env."""
    async with asyncio.timeout(5):
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False, trust_env=False) as client:
            async with client.stream("GET", url, headers={"Accept": "application/json"}) as response:
                response.raise_for_status()
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > 256_000:
                        raise ValueError("Public metadata response exceeds size limit")
                    chunks.append(chunk)
    value = json.loads(b"".join(chunks))
    if not isinstance(value, dict):
        raise ValueError("Public metadata must be an object")
    return value


class OAuthVerifier:
    def __init__(self, settings):
        self.settings = settings
        self._keys = {}
        self._expires = 0.0
        self._last_refresh = float("-inf")
        self._provider_failed = False
        self._lock = asyncio.Lock()

    async def _key(self, kid):
        async with self._lock:
            now = time.monotonic()
            if now < self._expires and kid in self._keys:
                return self._keys[kid]
            # Unknown kids cannot trigger an unbounded number of provider requests.
            if now - self._last_refresh < 10:
                if self._provider_failed:
                    raise AuthenticationError("AUTH_PROVIDER_UNAVAILABLE", "Authentication provider is temporarily unavailable.", 503)
                raise AuthenticationError("INVALID_TOKEN", "Access token could not be verified.")
            self._last_refresh = now
            try:
                data = await fetch_public_json(self.settings.jwks_url)
                candidates = data.get("keys")
                if not isinstance(candidates, list) or not 1 <= len(candidates) <= 50:
                    raise ValueError("Invalid signing keys")
                keys = {}
                for item in candidates:
                    if not isinstance(item, dict) or item.get("kty") != "RSA" or item.get("alg", "RS256") != "RS256":
                        continue
                    key_id = item.get("kid")
                    if (not isinstance(key_id, str) or not key_id or len(key_id) > 128
                            or item.get("use", "sig") != "sig" or "d" in item
                            or ("key_ops" in item and "verify" not in item["key_ops"])):
                        continue
                    key = jwt.PyJWK.from_dict(item, algorithm="RS256").key
                    if key.key_size < 2048 or key_id in keys:
                        raise ValueError("Invalid signing keys")
                    keys[key_id] = key
                if not keys:
                    raise ValueError("No signing keys")
                self._keys, self._expires = keys, now + 300
                self._provider_failed = False
            except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError, jwt.PyJWTError):
                self._provider_failed = True
                raise AuthenticationError("AUTH_PROVIDER_UNAVAILABLE", "Authentication provider is temporarily unavailable.", 503) from None
            if kid not in self._keys:
                raise AuthenticationError("INVALID_TOKEN", "Access token could not be verified.")
            return self._keys[kid]

    async def verify(self, token):
        try:
            if len(token) > 16_384:
                raise ValueError("Token too large")
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            if (header.get("alg") != "RS256" or header.get("typ", "JWT") not in {"JWT", "at+jwt"}
                    or not isinstance(kid, str) or not 1 <= len(kid) <= 128
                    or header.get("crit") or any(k in header for k in ("jku", "jwk", "x5u"))):
                raise ValueError("Unsupported token header")
            key = await self._key(kid)
            claims = jwt.decode(token, key, algorithms=["RS256"], issuer=self.settings.issuer,
                audience=self.settings.resource, leeway=30,
                options={"require": ["exp", "iat", "sub", "iss", "aud"], "strict_aud": False})
            # Auth0's RBAC dialect includes permissions; consented scopes alone
            # cannot grant a role permission the user does not actually possess.
            scope, permissions, subject = claims.get("scope", ""), claims.get("permissions"), claims.get("sub")
            if (not isinstance(scope, str) or not isinstance(permissions, list)
                    or not all(isinstance(p, str) for p in permissions)
                    or not isinstance(subject, str) or not subject or len(subject) > 255
                    or any(type(claims[k]) not in (int, float) for k in ("iat", "exp"))
                    or not 0 < claims["exp"] - claims["iat"] <= 3600):
                raise ValueError("Invalid access-token claims")
            azp, client_id = claims.get("azp"), claims.get("client_id")
            if azp is not None and client_id is not None and azp != client_id:
                raise ValueError("Conflicting client identity")
            client = azp or client_id
            if not isinstance(client, str) or not client or subject.endswith("@clients"):
                raise ValueError("An end-user access token is required")
        except (jwt.PyJWTError, ValueError, TypeError, OverflowError):
            raise AuthenticationError("INVALID_TOKEN", "Access token is invalid or expired.") from None
        if subject not in self.settings.allowed_subjects or (self.settings.allowed_clients and client not in self.settings.allowed_clients):
            raise AuthenticationError("FORBIDDEN", "This account or OAuth client is not permitted to access the platform.", 403)
        actor = "oauth:" + hashlib.sha256((self.settings.issuer + "\0" + subject).encode()).hexdigest()
        scopes = frozenset(scope.split()).intersection(permissions).intersection(SCOPES)
        return Principal(actor, scopes, "oauth")


class Authentication:
    def __init__(self):
        self.mode = os.getenv("ORDER_EXTRACTOR_MCP_AUTH_MODE", "bearer")
        self.settings = None
        self.verifier = None
        if self.mode == "oauth":
            try:
                self.settings = OAuthSettings.from_env()
                self.verifier = OAuthVerifier(self.settings)
            except ValueError:
                pass  # Fail closed at /mcp while leaving the ordinary API running.

    def ensure_ready(self):
        if (os.getenv("ORDER_EXTRACTOR_MCP_ENABLED") != "true"
                or self.mode not in {"bearer", "oauth"}
                or (self.mode == "oauth" and not self.settings)
                or (self.mode == "bearer" and len(os.getenv("ORDER_EXTRACTOR_MCP_TOKEN", "")) < 32)):
            raise AuthenticationError("MCP_UNAVAILABLE", "MCP authentication is not configured.", 503)

    async def authenticate(self, token):
        self.ensure_ready()
        if not token:
            raise AuthenticationError("UNAUTHENTICATED", "Sign in to access Order Extractor.")
        if self.mode == "oauth":
            return await self.verifier.verify(token)
        expected = os.getenv("ORDER_EXTRACTOR_MCP_TOKEN", "")
        if not hmac.compare_digest(token.encode(), expected.encode()):
            raise AuthenticationError("UNAUTHENTICATED", "Bearer authentication required.")
        return Principal("private-operator", frozenset(SCOPES), "bearer")

    def challenge(self, error=None, scopes=(READ_SCOPE,)):
        if self.settings:
            return self.settings.challenge(error=error, scopes=scopes)
        return 'Bearer realm="order-extractor-mcp"'


def require_scopes(principal, scopes):
    if not principal or not frozenset(scopes).issubset(principal.scopes):
        raise AuthenticationError("INSUFFICIENT_SCOPE", "Sign in with the required permissions for this action.", 403, tuple(scopes))
