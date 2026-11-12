import argparse
import logging
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser

import matplotlib.pyplot as plt
import requests

EXAMPLE_URL = (
    "https://zakon.rada.gov.ua/laws/show/"
    "254%D0%BA/96-%D0%B2%D1%80/print"
)
DEFAULT_TOP = 10
DEFAULT_WORKERS = 4
REQUEST_TIMEOUT = 30
# Some sites (e.g. Wikipedia) reject the default python-requests User-Agent.
HEADERS = {"User-Agent": "word-freq-mapreduce/1.0 (educational script)"}
# Tags whose content is not article text: code, styles and page navigation.
SKIPPED_TAGS = {
    "script", "style", "noscript", "template", "head",
    "nav", "header", "footer", "aside", "button", "form",
}
# CSS classes of service elements, e.g. Wikipedia's "[редагувати]" links.
SKIPPED_CLASSES = {"mw-editsection", "navbox", "reference", "mw-jump-link"}
# Tags that never have a closing tag.
VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "source", "track", "wbr",
}
# Letters only (any alphabet), allowing inner apostrophes, including the
# typographic ones used in Ukrainian: "don't", "ім’я", "пʼять".
WORD_PATTERN = re.compile(r"[^\W\d_]+(?:['’ʼ][^\W\d_]+)*")

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


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return number


class HTMLTextExtractor(HTMLParser):
    """Collect the readable text of an HTML page.

    Skips scripts, styles, navigation and other service elements.
    """

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        # Nesting depth inside a skipped element; 0 means "collect text".
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in VOID_TAGS:
            return
        if self._skip_depth:
            self._skip_depth += 1
            return
        classes = set((dict(attrs).get("class") or "").split())
        if tag in SKIPPED_TAGS or classes & SKIPPED_CLASSES:
            self._skip_depth = 1

    def handle_endtag(self, tag: str) -> None:
        if tag not in VOID_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)

    def get_text(self) -> str:
        return " ".join(self.parts)


def parse_args() -> argparse.Namespace:
    parser = HelpOnErrorParser(
        description=(
            "Analyze word frequency in a text from a URL with MapReduce."
        ),
        # argparse applies %-formatting to the epilog, so escape the URL.
        epilog=f"example: %(prog)s {EXAMPLE_URL.replace('%', '%%')} --top 10",
    )
    parser.add_argument("url", help="URL of a text or HTML page to analyze")
    parser.add_argument(
        "--top",
        type=positive_int,
        default=DEFAULT_TOP,
        help=f"number of top words to show (default: {DEFAULT_TOP})",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=DEFAULT_WORKERS,
        help=f"number of worker threads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        help="save the chart to FILE instead of showing it",
    )
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()
    return parser.parse_args()


def get_text(url: str) -> str:
    """Download the text from the given URL, stripping HTML markup if any."""
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "charset" not in content_type:
        # The server did not declare a charset; detect it from the content.
        response.encoding = response.apparent_encoding

    if "html" not in content_type:
        return response.text
    extractor = HTMLTextExtractor()
    extractor.feed(response.text)
    return extractor.get_text()


def split_into_words(text: str) -> list[str]:
    """Return lowercase words without punctuation or digits."""
    return WORD_PATTERN.findall(text.lower())


def split_into_chunks(words: list[str], chunks_count: int) -> list[list[str]]:
    """Split the word list into roughly equal chunks, one per worker."""
    chunk_size = max(1, -(-len(words) // chunks_count))  # ceiling division
    return [
        words[i:i + chunk_size] for i in range(0, len(words), chunk_size)
    ]


def map_function(words: list[str]) -> list[tuple[str, int]]:
    """Map phase: emit a (word, 1) pair for every word."""
    return [(word, 1) for word in words]


def shuffle_function(
    mapped_values: list[tuple[str, int]],
) -> dict[str, list[int]]:
    """Shuffle phase: group the emitted values by word."""
    shuffled = defaultdict(list)
    for key, value in mapped_values:
        shuffled[key].append(value)
    return shuffled


def reduce_function(key_values: tuple[str, list[int]]) -> tuple[str, int]:
    """Reduce phase: sum all values for a single word."""
    key, values = key_values
    return key, sum(values)


def map_reduce(text: str, workers: int = DEFAULT_WORKERS) -> dict[str, int]:
    """Count word frequencies, running map and reduce in a thread pool."""
    words = split_into_words(text)
    chunks = split_into_chunks(words, workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        mapped_chunks = executor.map(map_function, chunks)
        mapped_values = [pair for chunk in mapped_chunks for pair in chunk]

        shuffled_values = shuffle_function(mapped_values)

        reduced_values = executor.map(reduce_function, shuffled_values.items())
        return dict(reduced_values)


def visualize_top_words(
    word_counts: dict[str, int],
    top_n: int = DEFAULT_TOP,
    save_to: str | None = None,
) -> None:
    """Draw a horizontal bar chart of the most frequent words."""
    top_words = sorted(
        word_counts.items(), key=lambda item: item[1], reverse=True
    )[:top_n]
    words = [word for word, _ in top_words]
    counts = [count for _, count in top_words]

    plt.figure(figsize=(10, 6))
    plt.barh(words, counts, color="skyblue")
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.title(f"Top {len(words)} Most Frequent Words")
    plt.gca().invert_yaxis()  # the most frequent word on top
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to)
        logger.info("Chart saved to: %s", save_to)
    else:
        plt.show()


def main() -> None:
    args = parse_args()

    logger.info("Downloading text from: %s", args.url)
    try:
        text = get_text(args.url)
    except requests.RequestException as error:
        logger.error("Failed to download the text: %s", error)
        sys.exit(1)

    word_counts = map_reduce(text, args.workers)
    if not word_counts:
        logger.error("No words found in the text.")
        sys.exit(1)
    logger.info(
        "Found %d words, %d unique.",
        sum(word_counts.values()),
        len(word_counts),
    )

    visualize_top_words(word_counts, args.top, args.save)


if __name__ == "__main__":
    main()
