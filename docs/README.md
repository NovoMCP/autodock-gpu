# AutoDock-GPU Documentation

GPU-accelerated molecular docking service for the NovoMCP computational chemistry engine.

## Contents

- [API Reference](API_REFERENCE.md) — endpoints, request/response formats, authentication, and best practices for the retained docking API.

For an overview of the service, its tech stack, features, and how to run it, see the top-level [README](../README.md).

## Running

```
docker run -p 8022:8022 ghcr.io/novomcp/autodock-gpu:latest
```

Point the NovoMCP engine at this service by setting `AUTODOCK_GPU_URL` to its URL.
