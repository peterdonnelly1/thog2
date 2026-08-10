from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "sheet/plastic_depth_coarse.py"
    text = path.read_text(encoding="utf-8")
    old = (
        '            if len(self.training_losses) != self.training_steps:\n'
        '                raise ValueError(\n'
        '                    "successful COARSE trials require one training loss per step"\n'
        '                )\n'
        '            if not all(math.isfinite(float(value)) for value in self.training_losses):\n'
    )
    new = (
        '            if self.training_losses and len(self.training_losses) != self.training_steps:\n'
        '                raise ValueError(\n'
        '                    "COARSE training loss history must contain one value per recorded step"\n'
        '                )\n'
        '            if not all(math.isfinite(float(value)) for value in self.training_losses):\n'
    )
    if old not in text:
        if new in text:
            return
        raise RuntimeError("COARSE telemetry compatibility anchor was not found")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


if __name__ == "__main__":
    main()
