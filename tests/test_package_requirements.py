"""Published dependency markers must not exclude Windows ROCm 10's Torch."""

from importlib.metadata import requires

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.version import Version


def _windows_torch_requirements(extra: str) -> list[Requirement]:
    environment = default_environment()
    environment.update(sys_platform="win32", extra=extra)
    return [
        requirement
        for item in requires("piper-offload") or []
        if (requirement := Requirement(item)).name == "torch"
        and (requirement.marker is None or requirement.marker.evaluate(environment))
    ]


def test_windows_base_accepts_rocm10_torch() -> None:
    requirements = _windows_torch_requirements("")
    assert requirements
    assert all(Version("2.13.0+rocm10.0.0") in requirement.specifier for requirement in requirements)


def test_windows_triton38_requires_torch214() -> None:
    requirements = _windows_torch_requirements("triton")
    assert requirements
    assert any(Version("2.13.0+rocm10.0.0") not in requirement.specifier for requirement in requirements)
    assert all(Version("2.14.0") in requirement.specifier for requirement in requirements)
