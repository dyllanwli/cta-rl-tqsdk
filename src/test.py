import os 
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

import copy
import numpy as np
import modin.pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

from pytorch_forecasting.data.examples import get_stallion_data

# fix bug pytorch/30966
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from models.model_trainer import ModelTrainer
import pytz

MT = ModelTrainer()
data = MT.get_training_data()

def _datetime_to_timestamp_nano(dt: datetime) -> int:
    # timestamp() 返回值精度为 microsecond，直接乘以 1e9 可能有精度问题
    return int(dt.timestamp() * 1000000) * 1000

def _str_to_timestamp_nano(current_datetime: str, fmt="%Y-%m-%d %H:%M:%S.%f") -> int:
    return _datetime_to_timestamp_nano(datetime.strptime(current_datetime, fmt))

def _to_ns_timestamp(input_time):
    if type(input_time) in {int, float, np.float64, np.float32, np.int64, np.int32}:  # 时间戳
        if input_time > 2 ** 32:  # 纳秒( 将 > 2*32数值归为纳秒级)
            return int(input_time)
        else:  # 秒
            return int(input_time * 1e9)
    elif isinstance(input_time, str):  # str 类型时间
        return _str_to_timestamp_nano(input_time)
    elif isinstance(input_time, datetime):  # datetime 类型时间
        return _datetime_to_timestamp_nano(input_time)
    else:
        raise TypeError("暂不支持此类型的转换")

def time_to_s_timestamp(input_time):
    return int(_to_ns_timestamp(input_time) / 1e9)

def pre_process(data: pd.DataFrame):
    import time
    data = pd.DataFrame(data)
    exchange_tz = pytz.timezone('Asia/Shanghai')
    start = time.time()
    data["datetime"] =  data["datetime"].apply(lambda x: datetime.utcfromtimestamp(x.value / 1e9).astimezone(exchange_tz))
    data["time_idx"] = data.index.values
    data["is_daytime"] = data["datetime"].apply(lambda x: "1" if (x.hour >= 8 and x.hour < 16) else "0")
    end = time.time()
    print("Preprocess done", end - start)
    return data._to_pandas()

def build_dataloader(data: pd.DataFrame):
    max_prediction_length = 3
    max_encoder_length = 30
    training_cutoff = data["time_idx"].iloc[-1] - max_prediction_length
    variable_groups = {
        "time_variable": ["is_daytime"],
    }

    training = TimeSeriesDataSet(
        data[lambda x: x["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="close",
        group_ids=["instrument_id"],
        weight=None,
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["instrument_id"],
        static_reals=["duration"],
        time_varying_known_categoricals=["time_variable", "underlying_symbol"],
        variable_groups=variable_groups, 
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "open",
            "high",
            "low",
            "volume",
            "open_oi",
            "close_oi",
        ],
        target_normalizer=GroupNormalizer(
            groups=["instrument_id", "duration"], transformation="softplus"
        ),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    return training, validation

data = pre_process(data)
training, validation = build_dataloader(data)
# create dataloaders for model
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)


# calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader)
(actuals - baseline_predictions).abs().mean().item()

# configure network and trainer
pl.seed_everything(42)
trainer = pl.Trainer(
    gpus=0,
    # clipping gradients is a hyperparameter and important to prevent divergance
    # of the gradient for recurrent neural networks
    gradient_clip_val=0.1,
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    # not meaningful for finding the learning rate but otherwise very important
    learning_rate=0.03,
    hidden_size=16,  # most important hyperparameter apart from learning rate
    # number of attention heads. Set to up to 4 for large datasets
    attention_head_size=1,
    dropout=0.1,  # between 0.1 and 0.3 are good values
    hidden_continuous_size=8,  # set to <= hidden_size
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    # reduce learning rate if no improvement in validation loss after x epochs
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


# find optimal learning rate
res = trainer.tuner.lr_find(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    max_lr=10.0,
    min_lr=1e-6,
)

print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()


# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=30,
    gpus=1,
    enable_model_summary=True,
    gradient_clip_val=0.1,
    limit_train_batches=30,  # coment in for training, running valiation every 30 batches
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")