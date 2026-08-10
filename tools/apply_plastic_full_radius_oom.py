from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one full-radius OOM anchor, found {count}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def add_training_model_suffix_recovery() -> None:
    path = ROOT / "sheet/training_model.py"
    text = path.read_text(encoding="utf-8")
    marker = "    # ^^^ THOG\n\n    def forward(\n"
    class_start = text.index("class TrainingSheetGPT")
    marker_index = text.index(marker, class_start)
    method = '''    # vvv THOG full-radius CUDA probing retains every completed lower candidate when one upward suffix candidate OOMs
    def _plastic_depth_recoverable_probe_candidate_suffix(
        self,
        hidden: Tensor,
        targets: Tensor,
        request: PlasticDepthInlineProbeRequest,
    ) -> Tuple[Dict[int, Tensor], Tuple[Tuple[int, Tensor], ...], CheckpointExecutionReport]:
        upward_counts = tuple(int(value) for value in request.recoverable_upward_counts)
        prepare_upward = request.prepare_recoverable_upward_counts
        synchronize_candidate = request.synchronize_recoverable_upward_candidate
        if not upward_counts or prepare_upward is None or synchronize_candidate is None:
            raise RuntimeError("PLASTIC DEPTH full-radius recoverable probe request is incomplete")
        lower_counts = tuple(
            count for count in request.candidate_counts if count not in upward_counts
        )
        if not lower_counts or upward_counts[0] != lower_counts[-1] + 1:
            raise RuntimeError(
                "PLASTIC DEPTH recoverable upward suffix is not adjacent to the retained lower prefix"
            )
        lower_checkpoints, lower_report = execute_logical_layer_checkpoints(
            hidden,
            n_layer=self.config.n_layer,
            segment_size=self.checkpoint_segment_size,
            logical_block=self._logical_block,
            training=self.training,
            layer_indices=tuple(range(lower_counts[-1])),
            checkpoint_counts=lower_counts,
        )
        checkpoint_by_count = dict(lower_checkpoints)
        candidate_losses = []
        with torch.no_grad():
            for count in lower_counts:
                candidate_loss = self._plastic_depth_candidate_head_loss(
                    checkpoint_by_count[count],
                    targets,
                    request.sampled_token_indices,
                )
                candidate_losses.append((count, candidate_loss.detach()))

        prepare_upward()
        execution_report = lower_report
        for upward_count in upward_counts:
            prior_count = upward_count - 1
            prior_hidden = checkpoint_by_count.get(prior_count)
            if prior_hidden is None:
                raise RuntimeError(
                    "PLASTIC DEPTH upward suffix lost its preceding checkpoint; "
                    f"candidate={upward_count}, available={tuple(checkpoint_by_count)}"
                )
            upward_hidden: Optional[Tensor] = None
            upward_loss: Optional[Tensor] = None
            upward_report: Optional[CheckpointExecutionReport] = None
            local_feasible = True
            try:
                upward_hidden, upward_report = execute_logical_layers(
                    prior_hidden,
                    n_layer=self.config.n_layer,
                    segment_size=self.checkpoint_segment_size,
                    logical_block=self._logical_block,
                    training=self.training,
                    layer_indices=(upward_count - 1,),
                )
                with torch.no_grad():
                    upward_loss = self._plastic_depth_candidate_head_loss(
                        upward_hidden,
                        targets,
                        request.sampled_token_indices,
                    ).detach()
            except BaseException as error:
                if not is_cuda_out_of_memory(error):
                    raise
                local_feasible = False
                upward_hidden = None
                upward_loss = None
                upward_report = None
                if hidden.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            globally_feasible = bool(
                synchronize_candidate(upward_count, local_feasible)
            )
            if not globally_feasible:
                upward_hidden = None
                upward_loss = None
                upward_report = None
                # vvv THOG remote-rank rejection is not a local OOM and must not flush this rank's allocator cache
                if not local_feasible and hidden.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # ^^^ THOG
                break
            if upward_hidden is None or upward_loss is None or upward_report is None:
                raise RuntimeError(
                    "distributed PLASTIC DEPTH feasibility accepted a missing upward candidate"
                )
            checkpoint_by_count[upward_count] = upward_hidden
            candidate_losses.append((upward_count, upward_loss))
            execution_report = CheckpointExecutionReport(
                checkpointing_used=(
                    execution_report.checkpointing_used
                    or upward_report.checkpointing_used
                ),
                checkpoint_segments=(
                    execution_report.checkpoint_segments
                    + upward_report.checkpoint_segments
                ),
                logical_layers=(
                    execution_report.logical_layers
                    + upward_report.logical_layers
                ),
                segment_size=execution_report.segment_size,
            )
        return checkpoint_by_count, tuple(candidate_losses), execution_report
    # ^^^ THOG

'''
    replacement = "    # ^^^ THOG\n\n" + method + "    def forward(\n"
    text = text[:marker_index] + replacement + text[marker_index + len(marker):]
    path.write_text(text, encoding="utf-8")

    replace_once(
        "sheet/training_model.py",
        '            # vvv THOG preserve the exact established path unless the trainer explicitly supplies CUDA N+1 recovery coordination\n'
        '            if plastic_depth_probe_request.recoverable_upward_count is None:\n',
        '            # vvv THOG preserve the established path unless the trainer explicitly supplies recoverable CUDA upward candidates\n'
        '            if plastic_depth_probe_request.recoverable_upward_counts:\n'
        '                checkpoint_by_count, candidate_losses, self.last_execution_report = (\n'
        '                    self._plastic_depth_recoverable_probe_candidate_suffix(\n'
        '                        hidden,\n'
        '                        targets,\n'
        '                        plastic_depth_probe_request,\n'
        '                    )\n'
        '                )\n'
        '                checkpoints = tuple(checkpoint_by_count.items())\n'
        '            elif plastic_depth_probe_request.recoverable_upward_count is None:\n',
    )


def main() -> None:
    add_training_model_suffix_recovery()


if __name__ == "__main__":
    main()
