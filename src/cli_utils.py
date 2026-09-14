"""Small CLI validators that do not import the model runtime."""
import argparse
import math
from pathlib import Path
import tempfile


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('must be a nonnegative integer')
    return number


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError('must be finite')
    return number


def positive_float(value):
    number = finite_float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def nonnegative_float(value):
    number = finite_float(value)
    if number < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return number


def boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('expected true or false')


def require_file(value, label):
    if value is None or not str(value).strip():
        raise ValueError(f'{label} is required')
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'{label}: file does not exist: {path}')
    return path


def check_output_directory(path):
    """Check with a temporary file, leaving existing output files untouched."""
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=path):
        pass
    return path
