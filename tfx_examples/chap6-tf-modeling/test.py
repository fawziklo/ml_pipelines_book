import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np


ONE_HOT_FEATURES = {
    "product": None,
    "sub_product": None,
    "company_response": None,
    "state": None,
    "issue": None
}

# feature name, bucket count
BUCKET_FEATURES = {
    "zip_code": 10
}

# feature name, value is unused
TEXT_FEATURES = {
    "consumer_complaint_narrative": None
}

feature_names = ["product", "sub_product", "issue", "sub_issue", "state", "zip_code", "company", "company_response", "timely_response", "consumer_disputed", "consumer_complaint_narrative"]
df = pd.read_csv("http://bit.ly/building-ml-pipelines-dataset")
def make_one_hot(df):
    one_hot_array = []
    for feature_name in ONE_HOT_FEATURES.keys():
        temp_array = pd.np.asarray(tf.keras.utils.to_categorical(df[feature_name].values))
        ONE_HOT_FEATURES[feature_name] = temp_array.shape[1]
        one_hot_array.append(temp_array)

    return one_hot_array

def transformed_name(key):
    return key + '_xf'


def get_model(dp_optimizer, dp_loss, show_summary=True):
    print("in")
    """
    This function defines a Keras model and returns the model as a Keras object.
    """

    # one-hot categorical features
    input_features = []
    for key, dim in ONE_HOT_FEATURES.items():
        input_features.append(tf.keras.Input(shape=(dim), name=transformed_name(key)))

    # adding bucketized features
    for key, dim in BUCKET_FEATURES.items():
        input_features.append(tf.keras.Input(1, name=transformed_name(key)))

    # adding text input features
    input_texts = []
    for key in TEXT_FEATURES.keys():
        input_texts.append(tf.keras.Input(shape=(1,), name=transformed_name(key), dtype=tf.string))

    # embed text features
    print("la")
    MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.KerasLayer(MODULE_URL)
    print('lo')
    reshaped_narrative = tf.reshape(input_texts[0], [-1])
    embed_narrative = embed(reshaped_narrative)
    deep_ff = tf.keras.layers.Reshape((512,), input_shape=(1, 512))(embed_narrative)

    deep = tf.keras.layers.Dense(256, activation='relu')(deep_ff)
    deep = tf.keras.layers.Dense(64, activation='relu')(deep)
    deep = tf.keras.layers.Dense(16, activation='relu')(deep)

    wide_ff = tf.keras.layers.concatenate(input_features)
    wide = tf.keras.layers.Dense(16, activation='relu')(wide_ff)

    both = tf.keras.layers.concatenate([deep, wide])

    output = tf.keras.layers.Dense(1, activation='sigmoid')(both)

    inputs = input_features + input_texts

    keras_model = tf.keras.models.Model(inputs, output)
    keras_model.compile(optimizer=dp_optimizer,
                        loss=dp_loss,
                        metrics=[
                            tf.keras.metrics.BinaryAccuracy(),
                            tf.keras.metrics.TruePositives()
                        ])
    if show_summary:
        keras_model.summary()

    return keras_model

for feature in ONE_HOT_FEATURES.keys():
    df[feature] = df[feature].astype("category").cat.codes

one_hot_x = make_one_hot(df)

embedding_x = [pd.np.asarray(df[feature_name].values).reshape(-1) for feature_name in TEXT_FEATURES.keys()]

df['zip_code'] = df['zip_code'].str.replace('X', '0', regex=True)
df['zip_code'] = df['zip_code'].str.replace(r'\[|\*|\+|\-|`|\.|\ |\$|\/|!|\(', '0', regex=True)
df['zip_code'] = df['zip_code'].fillna(0)
df['zip_code'] = df['zip_code'].astype('int32')
# one bucket per 10k
df['zip_code'] = df['zip_code'].apply(lambda x: x//10000)
numeric_x = [df['zip_code'].values]

X = one_hot_x + numeric_x + embedding_x
y = np.asarray(df["consumer_disputed"], dtype=np.uint8).reshape(-1)




# DP parameters
NOISE_MULTIPLIER = 1.1
NUM_MICROBATCHES = 32
LEARNING_RATE = 0.1
POPULATION_SIZE = 1000
L2_NORM_CLIP = 1.0
BATCH_SIZE = 32
EPOCHS = 1

from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
optimizer = DPGradientDescentGaussianOptimizer(
    l2_norm_clip=L2_NORM_CLIP,
    noise_multiplier=NOISE_MULTIPLIER,
    num_microbatches=NUM_MICROBATCHES,
    learning_rate=LEARNING_RATE)

loss = tf.keras.losses.BinaryCrossentropy(
    from_logits=True, reduction=tf.losses.Reduction.NONE)




model = get_model(show_summary=True, dp_optimizer=optimizer, dp_loss=loss)
model.fit(x=X, y=y, batch_size=32, validation_split=0.1, epochs=EPOCHS, verbose=2)

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy


compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=POPULATION_SIZE,
                                              batch_size=BATCH_SIZE,
                                              noise_multiplier=NOISE_MULTIPLIER,
                                              epochs=EPOCHS,
                                              delta=1e-3)