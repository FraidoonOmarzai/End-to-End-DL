from zipfile import ZipFile
from PnemoniaDisease.utils.utils import read_yaml
from PnemoniaDisease import logger
import urllib.request as request
from PnemoniaDisease.constants import CONFIG_FILE_PATH
import os
from pathlib import Path
import argparse


STAGE_NAME = "Data Ingestion stage"

class DataIngestion:
    def __init__(self, params_filepath = CONFIG_FILE_PATH):
        self.config = read_yaml(params_filepath)


    def download_file(self):

        if not os.path.exists(self.config.data_ingestion.zip_dir):
            filename, headers = request.urlretrieve(url = self.config.data_ingestion.source_URL, 
                                                    filename = self.config.data_ingestion.zip_dir)
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info("File already exists")


    def extract_zip_file(self):
     
        with ZipFile(self.config.data_ingestion.zip_dir, "r") as zip:
            zip.extractall(self.config.data_ingestion.unzip_dir)
            logger.info("file extracted successfully")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=Path("config/config.yaml"))
    parsed_args = args.parse_args()

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_ingestion = DataIngestion(params_filepath=parsed_args.config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e