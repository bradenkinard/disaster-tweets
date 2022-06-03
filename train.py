import logging
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path='conf', config_name='config', version_base="1.2")
def main(cfg: DictConfig) -> None:
    preprocessor = instantiate(cfg.preprocessor)
    clf = instantiate(cfg.classifier)
    
    data = pd.read_csv('data/train.csv')
    train, test = train_test_split(data, random_state=1234)
    X_train, y_train = train['text'], train['target'].values
    X_test, y_test= test['text'], test['target'].values

    # Training
    X_train = preprocessor.fit_transform(X_train)
    clf.train(X_train, y_train)

    # Prediction
    X_test = preprocessor.transform(X_test)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)

    # Assessment
    logger.info(
        "Test accuracy: %s",
        accuracy_score(y_true=y_test, y_pred=y_pred)
    )
    logger.info(
        "Test ROC AOC Score: %s",
        roc_auc_score(y_true=y_test, y_score=y_pred_prob)
    )
    logger.info(
        "Test precision: %s",
        precision_score(y_true=y_test, y_pred=y_pred)
    )
    logger.info(
        "Test recall: %s",
        recall_score(y_true=y_test, y_pred=y_pred)
    )

if __name__ == "__main__":
    main()
