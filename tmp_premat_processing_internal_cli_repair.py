# vvv THOG
from pathlib import Path

path = Path("tmp_premat_processing_tranche.py")
text = path.read_text()

old_parser = '''    '    parser.add_argument("--premat_logging", choices=("enabled", "disabled"), default="disabled")\\n    parser.add_argument("--premat_instra", choices=("enabled", "disabled"), default="disabled")\\n    # vvv THOG exact public processing-diagnostic names; disabled/default frequency preserves existing runs\\n    parser.add_argument("--premat_processing_logging", choices=("enabled", "disabled"), default="disabled")\\n    parser.add_argument("--premat_processing_logging_capture_frequency_hz", type=int, default=10000)\\n    # ^^^ THOG\\n    parser.add_argument("--premat_retain_detailed_premat_history", type=_true_false, default=False, metavar="true|false")\\n',
'''
new_parser = '''    '    parser.add_argument("--premat_logging", choices=("enabled", "disabled"), default="disabled")\\n    parser.add_argument("--premat_instra", choices=("enabled", "disabled"), default="disabled")\\n    # vvv THOG public processing controls are consumed by the wrapper before core parsing; hidden aliases preserve the 15-option PREMAT scheduler surface\\n    parser.add_argument("--processing_logging_internal", dest="premat_processing_logging", choices=("enabled", "disabled"), default="disabled", help=argparse.SUPPRESS)\\n    parser.add_argument("--processing_logging_capture_frequency_hz_internal", dest="premat_processing_logging_capture_frequency_hz", type=int, default=10000, help=argparse.SUPPRESS)\\n    # ^^^ THOG\\n    parser.add_argument("--premat_retain_detailed_premat_history", type=_true_false, default=False, metavar="true|false")\\n',
'''
if text.count(old_parser) != 1:
    raise RuntimeError(f"expected one processing parser replacement body, found {text.count(old_parser)}")
text = text.replace(old_parser, new_parser, 1)

old_requested_tail = '''    validate_processing_configuration(logging, frequency, "cuda")
    return logging == "enabled", frequency


def _find_nsys()'''
new_requested_tail = '''    validate_processing_configuration(logging, frequency, "cuda")
    return logging == "enabled", frequency


def rewrite_processing_cli_for_core(arguments: Sequence[str]) -> list[str]:
    rewritten: list[str] = []
    index = 0
    replacements = {
        "--premat_processing_logging": "--processing_logging_internal",
        "--premat_processing_logging_capture_frequency_hz": "--processing_logging_capture_frequency_hz_internal",
    }
    while index < len(arguments):
        argument = str(arguments[index])
        matched = False
        for public_name, internal_name in replacements.items():
            if argument == public_name:
                if index + 1 >= len(arguments):
                    raise ValueError(f"{public_name} requires a value")
                rewritten.extend((internal_name, str(arguments[index + 1])))
                index += 2
                matched = True
                break
            prefix = public_name + "="
            if argument.startswith(prefix):
                rewritten.append(internal_name + "=" + argument[len(prefix):])
                index += 1
                matched = True
                break
        if matched:
            continue
        rewritten.append(argument)
        index += 1
    return rewritten


def _find_nsys()'''
if text.count(old_requested_tail) != 1:
    raise RuntimeError(f"expected one processing argv helper insertion point, found {text.count(old_requested_tail)}")
text = text.replace(old_requested_tail, new_requested_tail, 1)

old_reexec_head = '''def maybe_reexec_under_nsys(arguments: Sequence[str], *, entrypoint: Path) -> Optional[int]:
    requested, frequency = processing_requested_from_argv(arguments)
    if not requested or os.environ.get(_PROCESSING_CHILD_ENV) == "1":
        return None
    nsys = _find_nsys()'''
new_reexec_head = '''def maybe_reexec_under_nsys(arguments: Sequence[str], *, entrypoint: Path) -> Optional[int]:
    if os.environ.get(_PROCESSING_CHILD_ENV) == "1":
        return None
    requested, frequency = processing_requested_from_argv(arguments)
    rewritten_arguments = rewrite_processing_cli_for_core(arguments)
    if not requested:
        sys.argv[:] = [sys.argv[0], *rewritten_arguments]
        return None
    nsys = _find_nsys()'''
if text.count(old_reexec_head) != 1:
    raise RuntimeError(f"expected one nsys re-exec header, found {text.count(old_reexec_head)}")
text = text.replace(old_reexec_head, new_reexec_head, 1)

old_command_tail = '''        str(Path(entrypoint).resolve()),
        *arguments,
    ]'''
new_command_tail = '''        str(Path(entrypoint).resolve()),
        *rewritten_arguments,
    ]'''
if text.count(old_command_tail) != 1:
    raise RuntimeError(f"expected one nsys child argument tail, found {text.count(old_command_tail)}")
text = text.replace(old_command_tail, new_command_tail, 1)

old_export = '''    "processing_requested_from_argv",
    "register_processing_handoff",'''
new_export = '''    "processing_requested_from_argv",
    "rewrite_processing_cli_for_core",
    "register_processing_handoff",'''
if text.count(old_export) != 1:
    raise RuntimeError(f"expected one processing export block, found {text.count(old_export)}")
text = text.replace(old_export, new_export, 1)

old_test_import = '''    processing_requested_from_argv,
    validate_processing_configuration,
)'''
new_test_import = '''    processing_requested_from_argv,
    rewrite_processing_cli_for_core,
    validate_processing_configuration,
)'''
if text.count(old_test_import) != 1:
    raise RuntimeError(f"expected one processing test import block, found {text.count(old_test_import)}")
text = text.replace(old_test_import, new_test_import, 1)

old_test_anchor = '''def test_processing_configuration_is_cuda_but_not_premat_dependent() -> None:
'''
new_test_anchor = '''def test_processing_public_cli_rewrites_to_hidden_core_aliases() -> None:
    assert rewrite_processing_cli_for_core([
        "--premat_processing_logging", "enabled",
        "--premat_processing_logging_capture_frequency_hz=12345",
        "--model-type", "sheet",
    ]) == [
        "--processing_logging_internal", "enabled",
        "--processing_logging_capture_frequency_hz_internal=12345",
        "--model-type", "sheet",
    ]


def test_processing_configuration_is_cuda_but_not_premat_dependent() -> None:
'''
if text.count(old_test_anchor) != 1:
    raise RuntimeError(f"expected one processing test insertion point, found {text.count(old_test_anchor)}")
text = text.replace(old_test_anchor, new_test_anchor, 1)

path.write_text(text)
print("processing public-wrapper/internal-core CLI routing repaired")
# ^^^ THOG
