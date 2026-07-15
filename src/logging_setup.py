import logging


def get_logger(name: str, log_file: str = "outputs/pipeline.log") -> logging.Logger:
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)
