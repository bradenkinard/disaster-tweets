import logging
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    preprocessor = instantiate(cfg.preprocessor)
    clf = instantiate(cfg.classifier)
    
    data = pd.read_csv(get_original_cwd() + '/data/train.csv')
    train, test = train_test_split(data, random_state=1234)
    X_train, y_train = preprocessor.fit_transform(train), train['target'].values
    X_test, y_test = preprocessor.transform(test), test['target'].values

    # Training
    clf.train(X_train, y_train)

    # Prediction
    y_pred = clf.predict(X_test)

    # Assessment
    logger.info(
        "Model test accuracy: %s",
        accuracy_score(y_true=y_test, y_pred=y_pred)
    )


if __name__ == "__main__":
    main()
