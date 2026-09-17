"""Native registration instrumentation shared by host-memory integration tests."""

from piper_offload._host_registration import RuntimeHostRegistration


class RecordingBackend(RuntimeHostRegistration):
    def __init__(self) -> None:
        self.registrations: list[tuple[int, int]] = []
        self.unregistrations: list[int] = []

    def register(self, pointer: int, size: int) -> bool:
        registered = super().register(pointer, size)
        if registered:
            self.registrations.append((pointer, size))
        return registered

    def unregister(self, pointer: int) -> None:
        super().unregister(pointer)
        self.unregistrations.append(pointer)
