"""Asynchronously sort files from a source folder by their extensions.

Usage:
    python task1_sort.py <source> <output>
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import aiofiles
from aiopath import AsyncPath

CHUNK_SIZE = 64 * 1024
NO_EXTENSION_DIR = "no_extension"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class HelpOnErrorParser(argparse.ArgumentParser):
    """ArgumentParser that prints the full help on invalid arguments."""

    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"\nerror: {message}\n")


def parse_args() -> argparse.Namespace:
    parser = HelpOnErrorParser(
        description="Asynchronously sort files into subfolders by extension."
    )
    parser.add_argument("source", help="source folder to read files from")
    parser.add_argument("output", help="destination folder for sorted files")
    return parser.parse_args()


def validate_folders(source: Path, output: Path) -> None:
    """Exit with an error if folders are invalid for sorting."""
    if not source.is_dir():
        logger.error("Source folder does not exist: %s", source)
        sys.exit(1)
    if output == source or output.is_relative_to(source):
        logger.error(
            "Output folder %s cannot be inside the source folder %s. "
            "Please choose a destination outside of the source.",
            output,
            source,
        )
        sys.exit(1)


def build_name(file: AsyncPath, copy_number: int) -> str:
    """Return the file name, prefixed with Copy(n) for duplicates."""
    if copy_number == 0:
        return file.name
    return f"Copy({copy_number})_{file.name}"


async def copy_file(file: AsyncPath, output: AsyncPath) -> None:
    """Copy a file into the output subfolder named after its extension."""
    extension = file.suffix[1:].lower() or NO_EXTENSION_DIR
    target_dir = output / extension
    try:
        await target_dir.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(file, "rb") as src:
            copy_number = 0
            while True:
                target = target_dir / build_name(file, copy_number)
                try:
                    # "xb" fails if the file already exists, so concurrent
                    # copies with the same name never overwrite each other.
                    dst = await aiofiles.open(target, "xb")
                    break
                except FileExistsError:
                    copy_number += 1
            async with dst:
                while chunk := await src.read(CHUNK_SIZE):
                    await dst.write(chunk)
    except OSError as error:
        logger.error("Failed to copy %s: %s", file, error)


async def read_folder(source: AsyncPath, output: AsyncPath) -> None:
    """Recursively read the source folder and copy every file."""
    tasks = []
    try:
        async for item in source.iterdir():
            if await item.is_dir():
                tasks.append(read_folder(item, output))
            elif await item.is_file():
                tasks.append(copy_file(item, output))
    except OSError as error:
        logger.error("Failed to read folder %s: %s", source, error)
    await asyncio.gather(*tasks)


async def main() -> None:
    args = parse_args()
    source_path = Path(args.source).resolve()
    output_path = Path(args.output).resolve()
    validate_folders(source_path, output_path)

    source = AsyncPath(source_path)
    output = AsyncPath(output_path)
    await read_folder(source, output)
    logger.info("Sorting finished. Results are in: %s", output)


if __name__ == "__main__":
    asyncio.run(main())
