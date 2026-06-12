# Kubernetes Notes — Ingress

## Ingress configuration

On our staging cluster, **ingress** is handled by NGINX Ingress Controller v1.11.
TLS terminates at the ingress using cert-manager ClusterIssuer `letsencrypt-prod`.

## Path routing

`/api` routes to service `api-gateway` on port 8080.
`/static` routes to the CDN origin bucket via externalName service.
