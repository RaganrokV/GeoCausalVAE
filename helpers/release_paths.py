
import os
from pathlib import Path

INPUT_ROOT = Path(os.environ.get("BEEC_INPUT_ROOT", Path(__file__).resolve().parent / "input"))
OUTPUT_ROOT = Path(os.environ.get("BEEC_OUTPUT_ROOT", Path(__file__).resolve().parent / "generated_outputs"))

def output_path(relative):
    result = OUTPUT_ROOT / relative
    (result if relative.endswith(("/", "\\")) else result.parent).mkdir(parents=True, exist_ok=True)
    return str(result)
