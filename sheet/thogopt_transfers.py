# vvv THOG
"""Bounded asynchronous gradient staging and transient candidate budgeting."""
from __future__ import annotations

import torch


def candidate_memory_budget(*, allocated, reserved, peak, free, missing_state, workspace):
    """Reuse released peak headroom, or half the currently available memory.

    Cached allocator blocks are reusable; physical free memory alone understates
    availability. Reserving half the available memory permits small models to
    benefit too, without assigning every free byte to transaction candidates.
    State initialization uses host candidates so that persistent histories can
    be allocated separately from temporary backward/candidate storage. Aggregate
    free/cached bytes alone cannot establish that large blocks are reusable.
    """
    if missing_state:
        return 0
    available = max(0, free + reserved - allocated)
    released = max(0, peak - allocated)
    allowance = min(available, max(released, available // 2))
    return max(0, allowance - workspace)


class HostGradientTransfers:
    """Two pinned buffers; CPU consumers run only after their copy event finishes.

    Copies use a separate CUDA stream. record_stream protects the source storage
    after autograd releases it. Buffers are drained in submission order, including
    at the optimizer boundary. No additional dense gradient copy is retained.
    """
    def __init__(self, *, capacity, dtype, device):
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        self.buffers = [torch.empty(capacity, dtype=dtype, device="cpu", pin_memory=True) for _ in range(2)]
        self.events = [torch.cuda.Event() for _ in self.buffers]
        self.destinations = [None for _ in self.buffers]
        self.additions = [True for _ in self.buffers]
        self.next_slot = 0
        self.transfers = 0

    def _finish(self, slot, *, discard=False):
        destination = self.destinations[slot]
        if destination is None:
            return
        self.events[slot].synchronize()
        if not discard:
            values = self.buffers[slot][:destination.numel()].view_as(destination)
            if self.additions[slot]:
                destination.add_(values)
            else:
                destination.copy_(values)
        self.destinations[slot] = None

    @torch.no_grad()
    def enqueue(self, source, destination, *, add=True):
        if source.device != self.device or source.numel() != destination.numel():
            raise ValueError("thogopt gradient transfer device/shape mismatch")
        if source.numel() > self.buffers[0].numel():
            raise ValueError("thogopt gradient exceeds bounded transfer buffer")
        slot = self.next_slot
        self._finish(slot)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            # Protect original storage too: reshape can create a contiguous copy.
            source.record_stream(self.stream)
            contiguous = source.detach().reshape(-1)
            contiguous.record_stream(self.stream)
            self.buffers[slot][:source.numel()].copy_(contiguous, non_blocking=True)
            self.events[slot].record(self.stream)
        self.destinations[slot] = destination
        self.additions[slot] = add
        self.next_slot = (slot + 1) % len(self.buffers)
        self.transfers += 1

    def drain(self, *, discard=False):
        for offset in range(len(self.buffers)):
            self._finish((self.next_slot + offset) % len(self.buffers), discard=discard)

    @property
    def pinned_bytes(self):
        return sum(buffer.numel() * buffer.element_size() for buffer in self.buffers)
# ^^^ THOG
