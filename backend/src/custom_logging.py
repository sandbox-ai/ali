from datetime import datetime
import textwrap
import logging
import json
import os


def create_log_file(app_name: str, config: dict, log_directory: str):
    # Getting current system time and date
    current_time = datetime.now()

    # Create filepath:
    filepath = os.path.join(log_directory, current_time.strftime(f"%Y-%m-%d - %H-%M-%S - {app_name.upper()}.txt"))

    # Formatting the string with the current time and date
    formatted_message = current_time.strftime(f"%Y/%m/%d - %H:%M:%S - {app_name.upper()}\n")
    formatted_message += '-' * (len(formatted_message) - 1)
    formatted_message += '\n'

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Create log file:
    with open(filepath, 'w') as f:
        f.write(formatted_message)
        json.dump(config, f, indent=4)
        f.write('\n\n')

    return filepath


def save_user_message(user_message: str, filepath: str):
    formatted_message = textwrap.dedent(
        f"""\
            ========================================
            User: {user_message}

            """
    )

    # Save to file:
    with open(filepath, 'a') as f:
        f.write(formatted_message)


def save_bot_message(bot_message: str, bot_name: str, filepath: str, citations: dict = None):
    citation_text = ""
    if citations is not None:
        citation_text = "Sources:"
        for citation in citations.values():
            citation_text += f"\n\n{citation['score']*100:.2f}% - {citation['text']}\n{citation['metadata']}, {citation['start_char_idx']}-{citation['end_char_idx']}"

    formatted_message = (
f"""\
========================================
{bot_name}: {bot_message}

----------------------------------------
{citation_text}
"""
    )

    # Save to file:
    with open(filepath, 'a') as f:
        f.write(formatted_message)


def save_string(text: str, filepath: str):
    # Save to file:
    with open(filepath, 'a') as f:
        f.write(text)


class AnsiColorCodes:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ColorFormatter(logging.Formatter):
    FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

    COLOR_MAP = {
        logging.DEBUG: AnsiColorCodes.OKBLUE + FORMAT + AnsiColorCodes.ENDC,
        logging.INFO: AnsiColorCodes.OKGREEN + FORMAT + AnsiColorCodes.ENDC,
        logging.WARNING: AnsiColorCodes.WARNING + FORMAT + AnsiColorCodes.ENDC,
        logging.ERROR: AnsiColorCodes.FAIL + FORMAT + AnsiColorCodes.ENDC,
        logging.CRITICAL: AnsiColorCodes.FAIL + AnsiColorCodes.BOLD + FORMAT + AnsiColorCodes.ENDC,
    }

    def format(self, record):
        log_fmt = self.COLOR_MAP.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def set_up_logging():
    # Configure logging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter())

    # Replace the default handler with the colored one
    logging.getLogger().handlers = []
    logging.getLogger().addHandler(console_handler)