from sklearn.pipeline import Pipeline
from train.config import config
from train.processing import processing as pp


regression_pipeline = Pipeline(
    [
        ('normalization', pp.normalization(variable=config.NROM_var)),
        ('drop_values', pp.Drop_Features(variable=config.DROP_FEATURES)),
        ('add_delay', pp.Add_Delay(variable=config.DELAY_parm))

    ]



)