import os
from typing import List, Tuple, Union
import logging

from langchain.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharactorSplitter
from langchain.vectorstores import Chroma

from consts import CHROMA_SETTINGS, DOCUMENT_MAP, PERSIST_DIRECTORY, SOURCE_DIRECTORY
import argparse

logging = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)

args = argparse.ArgumentParser()
args.add_argument("--source_dir", type=str, default=SOURCE_DIRECTORY)
args.add_argument("--persist_dir", type=str, default=PERSIST_DIRECTORY)
args.add_argument(
    "--device_type",
    type=str,
    default="cuda",
    choices=["cuda", "cpu", "hip", "xla", "ort", "tpu", "mkldnn"],
    help="The compute power that you have",
)


def load_single_document(file_path: str) -> Document:
    """
    Load one document from the source documents directory
    """

    file_extension = os.path.splitext(file_path)[1]
    # ingestor = Ingestor(file_extension)
    try:
        loader_class = DOCUMENT_MAP[file_extension]
    except KeyError:
        raise KeyError(f"File extension {file_extension} is not supported")
    finally:
        pass

    loader = loader_class(file_path)

    return loader.load()[0]


def load_document(source_dir: str) -> List[Document]:
    """
    Load all documents from the source documents directory
    """

    all_files = os.listdir(source_dir)
    documents = []

    for file in all_files:
        source_file_path = os.path.join(source_dir, file)
        documents.append(load_single_document(source_file_path))

    return documents


def main():
    args = args.parse_args()

    logging.info("Loading documents from {args.source_dir}")
    documents = load_document(args.source_dir)

    logging.info("Splitting documents into chunks and processing text")
    text_splitter = RecursiveCharactorSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    logging.info("Creating embedding for the documents")
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": args.device_type}
    )
    # Create a vector store
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=args.persist_dir,
        client_settings=CHROMA_SETTINGS,
    )

    db.persist()
    db = None


if __name__ == "__main__":
    main()
