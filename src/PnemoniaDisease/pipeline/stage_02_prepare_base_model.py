from PnemoniaDisease.utils.utils import read_yaml, create_directories
from PnemoniaDisease import logger
from PnemoniaDisease.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from pathlib import Path
import argparse
import tensorflow as tf


STAGE_NAME = "Prepare Base Model"

class PrepareBaseModel:
    def __init__(self,
                 params_filepath=PARAMS_FILE_PATH,
                 config_filepath=CONFIG_FILE_PATH):
        self.params = read_yaml(params_filepath)
        self.config = read_yaml(config_filepath)

        create_directories([self.config.artifacts_root])
        create_directories([self.config.prepare_base_model.root_dir])


    def base_model(self):
        model_base = tf.keras.applications.EfficientNetB0(
            input_shape=self.params.IMAGE_SIZE,
            weights=self.params.WEIGHTS,
            include_top=self.params.INCLUDE_TOP
        )
        return model_base


    @staticmethod
    def full_model(model_base, classes, freeze_all, freeze_till, learning_rate):

        if freeze_all:
            for layer in model_base.layers:
                model_base.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model_base.layers[:-freeze_till]:
                model_base.trainable = False
        
        x = model_base.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x) # pool the outputs of the base model
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(model_base.input, outputs)

        
        # Compile
        model.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(learning_rate),
                    metrics=["accuracy"])
        
        model.summary()
        return model

    def update_base_model(self):
        self.full_model = self.full_model(
            model_base=self.base_model(),
            classes=self.params.CLASSES,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.params.LEARNING_RATE
        )

        self.save_model(
            path=self.config.prepare_base_model.base_model_path, 
            model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=[Path("config/config.yaml"), Path("params.yaml")])
    parsed_args = args.parse_args()

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        prepare_base_model = PrepareBaseModel(params_filepath=PARAMS_FILE_PATH,
                                              config_filepath=CONFIG_FILE_PATH)
        prepare_base_model.update_base_model()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e