
import os

import pandas as _pandas

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    JSONLoader,
    Docx2txtLoader,
    BSHTMLLoader,
    DataFrameLoader,
)


class UnsupportedFileTypeError(Exception):
    pass


# ------------------------------------------------------------------
# SUPPORTED FILE TYPES
#
# ext   -> loader class + kwargs
# ------------------------------------------------------------------

_LOADERS = {
    "pdf": {
        "loader": PyPDFLoader,
        "kwargs": {}
    },
    "txt": {
        "loader": TextLoader,
        "kwargs": {}
    },
    "md": {
        "loader": TextLoader,
        "kwargs": {}
    },
    "csv": {
        "loader": CSVLoader,
        "kwargs": {}
    },
    "json": {
        "loader": JSONLoader,
        "kwargs": {
            "jq_schema": ".[]"
        }
    },
    "docx": {
        "loader": Docx2txtLoader,
        "kwargs": {}
    },
    "html": {
        "loader": BSHTMLLoader,
        "kwargs": {}
    },
    "htm": {
        "loader": BSHTMLLoader,
        "kwargs": {}
    },
}


def get_file_extension(file_path: str) -> str:

    return os.path.splitext(
        file_path
    )[1].lower().lstrip(".")


def supported_extensions() -> list:

    return sorted(
        _LOADERS.keys()
    )


def get_loader(
    file_path: str
):

    extension = get_file_extension(
        file_path
    )

    if extension == "xlsx":

        data_frame = _pandas.read_excel(
            file_path
        )

        return DataFrameLoader(
            data_frame,
            page_content_column="text"
        )

    config = _LOADERS.get(
        extension
    )

    if config is None:

        raise UnsupportedFileTypeError(
            f"Unsupported file type: .{extension}. "
            f"Supported types are: "
            f"{', '.join(supported_extensions())}, xlsx"
        )

    loader_class = config["loader"]

    kwargs = config["kwargs"]

    return loader_class(
        file_path,
        **kwargs
    )


def load_document(
    file_path: str
):

    loader = get_loader(
        file_path
    )

    documents = loader.load()

    return documents
