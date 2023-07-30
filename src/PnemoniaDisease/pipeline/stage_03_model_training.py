from PnemoniaDisease.utils.utils import read_yaml, create_directories
from PnemoniaDisease import logger
from PnemoniaDisease.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from pathlib import Path
import argparse
import tensorflow as tf
import os
import time

STAGE_NAME = "Model Training"

class PrepareCallback:
    def __init__(self,config_filepath=CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)      
    
    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.prepare_callbacks.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        logger.info("TensorBoard created")
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    

    @property
    def _create_ckpt_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.prepare_callbacks.checkpoint_model_filepath,
            save_best_only=True
        )
        logger.info("Checkpoint created")


    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]
    

class Training(PrepareCallback):
    def __init__(self,
                 params_filepath=PARAMS_FILE_PATH,
                 config_filepath=CONFIG_FILE_PATH):
        super().__init__(params_filepath)
        self.params = read_yaml(params_filepath)
        self.config = read_yaml(config_filepath)

        create_directories([self.config.artifacts_root])

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.prepare_base_model.base_model_path
        )
    
    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.params.IMAGE_SIZE[:-1],
            batch_size=self.params.BATCH_SIZE,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        training = self.config.training
        create_directories([
            Path(training.root_dir)
        ])
        training_data = os.path.join("data/processed", "train")
        

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=Path(training_data),
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.params.AUGMENTATION:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
            logger.info("data augmentation part")
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=Path(training_data),
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        logger.info("model saved")


    def train(self, callback_list: list): 
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.params.EPOCHS,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )
        logger.info("model trained")

        self.save_model(
            path=self.config.training.trained_model_path,
            model=self.model
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=[Path("config/config.yaml"), Path("params.yaml")])
    parsed_args = args.parse_args()

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        prepare_callbacks = PrepareCallback(config_filepath=CONFIG_FILE_PATH)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        training = Training(config_filepath=CONFIG_FILE_PATH, 
                            params_filepath=PARAMS_FILE_PATH)
        training.get_base_model()
        training.train_valid_generator()
        training.train(
            callback_list=callback_list
        )
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e