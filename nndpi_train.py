import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras import callbacks
from tensorflow.keras.layers import (
    GRU,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Embedding,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def create_model(max_len, dropout_rate=0.05):

    inp = Input(shape=(max_len))
    x = Embedding(np.iinfo(np.uint8).max + 1, 1, input_length=max_len)(inp)
    x = Conv1D(
        filters=64, kernel_size=25, strides=1, padding="valid", activation="relu"
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=4)(x)
    x = Conv1D(
        filters=128, kernel_size=16, strides=1, padding="valid", activation="relu"
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Conv1D(
        filters=256, kernel_size=5, strides=1, padding="valid", activation="relu"
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(
        filters=512, kernel_size=2, strides=1, padding="valid", activation="relu"
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Bidirectional(
        GRU(units=64, dropout=dropout_rate, recurrent_dropout=dropout_rate)
    )(x)
    x = Dense(64, activation="relu", kernel_initializer="glorot_normal")(x)
    x = BatchNormalization()(x)

    out1 = Dense(n_tag_1, activation="softmax", name="tag_1")(x)
    out2 = Dense(n_tag_2, activation="softmax", name="tag_2")(x)
    out3 = Dense(n_tag_3, activation="softmax", name="tag_3")(x)
    out4 = Dense(n_tag_4, activation="softmax", name="tag_4")(x)

    return Model(inputs=inp, outputs=[out1, out2, out3, out4])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multi_gpu", default=True, type=bool, help="Train on multi gpu system"
    )
    parser.add_argument(
        "--batch_size", default=3072, type=np.uint32, help="Training batch_size"
    )
    parser.add_argument(
        "--max_len",
        default=1500,
        type=np.uint32,
        help="Preprocessed packet length in bytes",
    )

    parser.add_argument(
        "--feather_df",
        default="./CombinedPackets/allpkts.feather",
        type=str,
        help="Path to combined packets feather dataframe",
    )
    args = parser.parse_args()

    print("Reading preprocessed packets..")
    allpkts = pd.read_feather(args.feather_df, columns=[str(i) for i in range(1500)])
    meta = pd.read_feather(
        args.feather_df,
        columns=["tag_1", "tag_2", "tag_3", "tag_4", "protocol", "filename", "ix"],
    )

    print("Encoding labels..")
    le_1, le_2, le_3, le_4 = (
        LabelEncoder(),
        LabelEncoder(),
        LabelEncoder(),
        LabelEncoder(),
    )
    tag_1 = le_1.fit_transform(meta.tag_1)
    tag_2 = le_2.fit_transform(meta.tag_2)
    tag_3 = le_3.fit_transform(meta.tag_3)
    tag_4 = le_4.fit_transform(meta.tag_4)

    n_tag_1 = len(np.unique(tag_1))
    n_tag_2 = len(np.unique(tag_2))
    n_tag_3 = len(np.unique(tag_3))
    n_tag_4 = len(np.unique(tag_4))

    print("Creating stratified train/validation/test splits (80%, 10%, 10%)..")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=555)
    train_index, test_valid_index = next(sss.split(allpkts, meta.filename))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=555)
    test_index, val_index = next(
        sss.split(allpkts.iloc[test_valid_index], meta.filename.iloc[test_valid_index])
    )
    test_index = allpkts.iloc[test_valid_index].iloc[test_index].index
    val_index = allpkts.iloc[test_valid_index].iloc[val_index].index

    print("Calculating class weigths..")
    tag_1_class_weights = class_weight.compute_class_weight(
        "balanced", np.unique(tag_1), tag_1
    )
    tag_2_class_weights = class_weight.compute_class_weight(
        "balanced", np.unique(tag_2), tag_2
    )
    tag_3_class_weights = class_weight.compute_class_weight(
        "balanced", np.unique(tag_3), tag_3
    )
    tag_4_class_weights = class_weight.compute_class_weight(
        "balanced", np.unique(tag_4), tag_4
    )

    losses = {
        "tag_1": "sparse_categorical_crossentropy",
        "tag_2": "sparse_categorical_crossentropy",
        "tag_3": "sparse_categorical_crossentropy",
        "tag_4": "sparse_categorical_crossentropy",
    }

    class_weights = {
        "tag_1": tag_1_class_weights,
        "tag_2": tag_2_class_weights,
        "tag_3": tag_3_class_weights,
        "tag_4": tag_4_class_weights,
    }

    es = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=4,
        verbose=1,
        mode="min",
        baseline=None,
        restore_best_weights=True,
    )

    cp = callbacks.ModelCheckpoint(
        "dpi.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_weights_only=False,
    )

    rlr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, mode="min", verbose=1
    )

    csv_logger = callbacks.CSVLogger("history.csv", separator=",", append=True)

    if args.multi_gpu:
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
        )
        with strategy.scope():
            model = create_model(max_len=args.max_len)
            model.compile(loss=losses, optimizer=Adam(), metrics=["accuracy"])
    else:
        model = create_model(max_len=args.max_len)
        model.compile(loss=losses, optimizer=Adam(), metrics=["accuracy"])

    print("Training starting...")
    model.fit(
        x=allpkts.iloc[train_index].to_numpy(),
        y={
            "tag_1": tag_1[train_index],
            "tag_2": tag_2[train_index],
            "tag_3": tag_3[train_index],
            "tag_4": tag_4[train_index],
        },
        validation_data=(
            allpkts.iloc[val_index].to_numpy(),
            {
                "tag_1": tag_1[val_index],
                "tag_2": tag_2[val_index],
                "tag_3": tag_3[val_index],
                "tag_4": tag_4[val_index],
            },
        ),
        class_weight=class_weights,
        batch_size=args.batch_size,
        callbacks=[es, cp, csv_logger, rlr],
        epochs=100,
    )

