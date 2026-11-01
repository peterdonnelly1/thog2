import errno
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from sheet.checkpoints import save_payload


def test_enospc_preserves_previous_complete_checkpoint(tmp_path):
    target=tmp_path/'ckpt.pt'
    save_payload({'value':1},target)
    original=target.read_bytes()
    def fail(payload,stream):
        stream.write(b'partial')
        raise OSError(errno.ENOSPC,'No space left on device')
    with patch.object(torch,'save',side_effect=fail), pytest.raises(OSError):
        save_payload({'value':2},target)
    assert target.read_bytes()==original
    assert list(tmp_path.glob('*.tmp'))==[]


def test_atomic_replacement_preserves_linked_bundle(tmp_path):
    import os
    target=save_payload({'value':1},tmp_path/'ckpt.pt')
    bundle=tmp_path/'bundle.pt';os.link(target,bundle)
    assert bundle.stat().st_ino==target.stat().st_ino
    save_payload({'value':2},target)
    assert torch.load(bundle,weights_only=False)['value']==1
    assert torch.load(target,weights_only=False)['value']==2
    assert bundle.stat().st_ino!=target.stat().st_ino
