from pathlib import Path
from PnemoniaDisease.utils.utils import read_yaml, save_json
from PnemoniaDisease.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from PnemoniaDisease import logger
import tensorflow as tf
import os


STAGE_NAME = "Model Evaluations"
class Evaluation:
    def __init__(self, config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):
        
        self.params = read_yaml(params_filepath)
        self.config = read_yaml(config_filepath)
    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.params.IMAGE_SIZE[:-1],
            batch_size=self.params.BATCH_SIZE,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        training_data = os.path.join("data/processed", "train")
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=Path(training_data),
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.training.trained_model_path)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)

    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)



if __name__ == "__main__":

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_eval = Evaluation(config_filepath=CONFIG_FILE_PATH,
                                params_filepath=PARAMS_FILE_PATH)
        model_eval.evaluation()
        model_eval.save_score()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e