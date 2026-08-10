# Versioning and releases

Piper Offload uses Python's PEP 440 version format and `v<version>` Git tags. The version in
`pyproject.toml`, the Git tag, the PyPI release, and the GitHub release must agree.

## Compatibility policy

The package is pre-1.0 while its memory-management and adapter interfaces are being established:

- Patch releases (`0.1.1`) preserve existing documented behavior and contain fixes,
  compatible performance improvements, or backward-compatible capabilities.
- Minor releases (`0.2.0`) may add larger capabilities or deliberately change a public
  pre-1.0 contract.
- Release candidates (`0.2.0rc1`) are used when a change needs integration testing before a
  stable release.

After 1.0, patch releases contain compatible fixes, minor releases add backward-compatible
features, and major releases contain breaking changes.

Public compatibility includes exported names, constructor and method signatures, resource and
activation lifecycle behavior, tensor-adapter registration, documented cache and streaming
semantics, and supported package extras. Modules and names beginning with an underscore are
internal.

## Dependency policy

Piper Offload declares the PyTorch, Python, and optional-backend versions that each release is
tested against. Patch releases may widen compatible dependency ranges or raise an optional
backend's minimum when needed by a compatible fix or capability. Raising a core Python or
PyTorch minimum, or dropping a supported platform or dependency line, requires at least a minor
release while the package is pre-1.0.

## Release process

1. Start from a green `main` branch and choose the next version from the policy above.
2. Run `uv version <version>` in a release branch and move the `Unreleased` changelog entries
   under a dated heading for that version.
3. Open and merge the release pull request after CPU and GPU validation passes.
4. Create and push an annotated `v<version>` tag on the merge commit.
5. Review and approve the protected `pypi` GitHub environment deployment.
6. Verify the PyPI installation and the automatically created GitHub release.

The release workflow builds each distribution once, validates the wheel in an isolated
environment, publishes those exact artifacts to PyPI through Trusted Publishing, and attaches
the same files to the GitHub release. Published versions and artifacts are immutable; fixes use
a new version.
