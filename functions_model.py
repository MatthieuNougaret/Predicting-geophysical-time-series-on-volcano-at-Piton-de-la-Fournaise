import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from pathlib import Path
import joblib

from functions_plot import *
from functions_metrics import *

def train_models(X_train, y_train, 
                 X_valid, y_valid, 
                 args,
                 features, 
                 output_path):
    """
    Function to evaluates models and saves training history.

    Parameters
    ----------
    X_train : numpy.ndarray
        3d tensor of the training dataset input.
    y_train : numpy.ndarray
        3d tensor of the training dataset output.
    X_valid : numpy.ndarray
        3d tensor of the validation dataset input.
    y_valid : numpy.ndarray
        3d tensor of the validation dataset output.
    X_test : numpy.ndarray
        3d tensor of the testing dataset input.
    y_test : numpy.ndarray
        3d tensor of the testing dataset output.
    args : dict
        Dictionary with init file informations.
    features : numpy.ndarray
        1d vector with all the features.
    output_path : pathlib.Path
        Where to save the models.

    Returns
    -------
    performance_df : pandas.DataFrame
        Data Frame with the models score results.

    """
    num_features = len(features)
    
    # Define models
    models = {
        'Last': Baseline(args['OUT_STEPS']),
        'Linear': build_linear_model(args, num_features),
        'Dense': build_dense_model(args, num_features),
        'Conv1D': build_conv_model(args, num_features),
        'LSTM': build_lstm_model(args, num_features),
        'Transformer': build_autoregressive_transformer(input_shape=(args['INPUT_WIDTH'], num_features), #(timesteps, features),
                                               head_size=args['head_size'], 
                                               num_heads=args['num_heads'],
                                               ff_dim=args['ff_dim'],
                                               num_transformer_blocks=args['num_transformer_blocks'], 
                                               mlp_units=[int(u.strip()) for u in args["mlp_units"].split(",")],
                                               out_steps=args['OUT_STEPS'],
                                               mlp_dropout=args['mlp_dropout'], # mlp_dropout
                                               norm_type=args['norm_type'], # "post" or "pre" to try pre-norm => post better
                                              ),

    }

    learning_rates = {
        'Last': 0.001,
        'Linear': args['LR_Linear'],
        'Dense': args['LR_Dense'],
        'Conv1D': args['LR_Conv'],
        'LSTM': args['LR_LSTM'],
        'Transformer': args['LR_Transformer'],
    }

    for idx, (name, model) in enumerate(models.items()):
        print("----------------------------------")
        print("\n---- Training model {} ----".format(name))

        # Train and evaluate
        if name != 'Last':
            history = compile_and_fit(
                model, 
                (X_train, y_train), 
                (X_valid, y_valid),
                patience=args['PATIENCE'], 
                save_folder=output_path / name,
                max_epoch=args['MAX_EPOCHS'], 
                batch_size=args['BATCH_SIZE'],
                start_lr=learning_rates[name], 
                lr_factor=args['LR_FACTOR'],
                warmup_epochs=10,
                huber_delta=args['huber_delta']
            )

            plot_learning(history.history, y_scale='log', save_p=output_path / name)

def evaluate_models(X_train, y_train, 
                    X_valid, y_valid, 
                    X_test, y_test, 
                    args,
                    features, 
                    output_path):
    """
    Evaluate models and save training history.
    Reports R², MASE, and peak-weighted RMSE.
    """
    num_features = len(features)

    model_names = ['Last', 'Linear', 'Dense', 
                   'Conv1D', 'LSTM', 'Transformer']

    # --- prepare the output dataframe ---
    performance_df = pd.DataFrame(model_names, columns=["Model"])

    performance_df['Valid_MASE'], performance_df['Valid_pRMSE'] = 0.0, 0.0
    performance_df['Test_MASE'], performance_df['Test_pRMSE'] = 0.0, 0.0
    performance_df['Valid_R2'], performance_df['Test_R2'] = 0.0, 0.0

    # --- add per-feature R2 ---
    for i in range(num_features):
        performance_df['Valid_R2_'+features[i]] = 0.0
        performance_df['Test_R2_'+features[i]] = 0.0

    # --- add per-day R2 ---
    for i in range(args['OUT_STEPS']):
        performance_df['Valid_R2_t+'+str(i+1)] = 0.0
        performance_df['Test_R2_t+'+str(i+1)] = 0.0

    # --- Loop on the models 
    for idx, name in enumerate(model_names):
        # --- load model ---
        if name == "Last":
            model = Baseline(args['OUT_STEPS'])
        else:
            model = keras.models.load_model(output_path / name / 'best_model.keras')
        
        # --- make predictions ---
        train_preds_sc = model(X_train, training=False).numpy()
        valid_preds_sc = model(X_valid, training=False).numpy()
        test_preds_sc  = model(X_test, training=False).numpy()

        # --- load scaler and inverse transform ---
        PATH_SCALER = Path(args['PATH'])
        scaler = joblib.load(PATH_SCALER / 'scaler.pkl')

        y_train_inv = scaler.inverse_transform(y_train)
        y_valid_inv = scaler.inverse_transform(y_valid)
        y_test_inv  = scaler.inverse_transform(y_test)

        train_preds = scaler.inverse_transform(train_preds_sc)
        valid_preds = scaler.inverse_transform(valid_preds_sc)
        test_preds  = scaler.inverse_transform(test_preds_sc)

        # --- compute metrics ---
        def compute_metrics(y_true, y_pred):
            r2   = r2_score(y_true.ravel(), y_pred.ravel())
            mase_val = mase(y_true, y_pred)
            prmse = peak_weighted_rmse(y_true, y_pred, q=0.9, alpha=4.0)
            return r2, mase_val, prmse

        train_r2, train_mase, train_prmse = compute_metrics(y_train_inv, train_preds)
        valid_r2, valid_mase, valid_prmse = compute_metrics(y_valid_inv, valid_preds)
        test_r2,  test_mase,  test_prmse  = compute_metrics(y_test_inv,  test_preds)

        # --- logging ---
        print(f"Metrics for model {name}")
        print(f"Train MASE {train_mase:.4f} | pRMSE {train_prmse:.4f} | R2 {train_r2:.3f}")
        print(f"Valid MASE {valid_mase:.4f} | pRMSE {valid_prmse:.4f} | R2 {valid_r2:.3f}")

        # --- save global metrics ---
        performance_df.loc[idx, 'Train_MASE'] = train_mase
        performance_df.loc[idx, 'Train_pRMSE'] = train_prmse
        performance_df.loc[idx, 'Train_R2'] = train_r2

        performance_df.loc[idx, 'Valid_MASE'] = valid_mase
        performance_df.loc[idx, 'Valid_pRMSE'] = valid_prmse
        performance_df.loc[idx, 'Valid_R2'] = valid_r2

        performance_df.loc[idx, 'Test_MASE'] = test_mase
        performance_df.loc[idx, 'Test_pRMSE'] = test_prmse
        performance_df.loc[idx, 'Test_R2'] = test_r2

        # --- per feature R2 ---
        for i in range(num_features):
            yt = y_valid_inv[:,:,i].ravel()
            yp = valid_preds[:,:,i].ravel()
            performance_df.loc[idx, 'Valid_R2_'+features[i]] = r2_score(yt, yp)

            yt = y_test_inv[:,:,i].ravel()
            yp = test_preds[:,:,i].ravel()
            performance_df.loc[idx, 'Test_R2_'+features[i]] = r2_score(yt, yp)

        # --- per day R2 ---
        for i in range(args['OUT_STEPS']):
            yt = y_valid_inv[:,i,:].ravel()
            yp = valid_preds[:,i,:].ravel()
            performance_df.loc[idx, 'Valid_R2_t+'+str(i+1)] = r2_score(yt, yp)

            yt = y_test_inv[:,i,:].ravel()
            yp = test_preds[:,i,:].ravel()
            performance_df.loc[idx, 'Test_R2_t+'+str(i+1)] = r2_score(yt, yp)

    return performance_df

class Baseline(tf.keras.Model):
    """
    Build Last method. It repeat the last seen value.
    """

    def __init__(self, OUT_STEPS):
        """
        Initialise Last method.

        Parameters
        ----------
        OUT_STEPS : int
            Number of times the last value is repeated.

        Returns
        -------
        None

        """
        super(Baseline, self).__init__()
        self.OUT_STEPS = OUT_STEPS
    
    def call(self, inputs):
        """
        Prediction method.

        Parameters
        ----------
        inputs : numpy.ndarray
            3d tensor, inpunt data.

        Returns
        -------
        tf.tile
            Model prediction.

        """
        return tf.tile(inputs[:, -1:, :], [1, self.OUT_STEPS, 1])
    
def build_linear_model(args, num_features):
    """
    Linear model.

    Parameters
    ----------
    args : dict
        Dictionary of parameters.
    num_features : int
        Number of features.

    Returns
    -------
    tensorflow.keras model

    """
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(args['OUT_STEPS'] * num_features, 
                              kernel_initializer='glorot_uniform'),
        tf.keras.layers.Reshape([args['OUT_STEPS'], num_features])
    ])

def build_dense_model(args, num_features):
    """
    Build a Dense model.

    Parameters
    ----------
    args : dict
        Dictionary of parameters.
    num_features : int
        Number of features.

    Returns
    -------
    tensorflow.keras model
        Dense model.

    """
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(args['DENSE_UNITS'], 
                              activation='swish', 
                              kernel_initializer='glorot_uniform',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.Dense(args['OUT_STEPS'] * num_features, 
                              kernel_initializer='glorot_uniform',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.Reshape([args['OUT_STEPS'], num_features])
    ])

def build_conv_model(args, num_features):
    """
    Build a 1d Convolutional model.

    Parameters
    ----------
    args : dict
        Dictionary of parameters.
    num_features : int
        Number of features.

    Returns
    -------
    tensorflow.keras model
        1d Convolutional model.

    """
    return tf.keras.Sequential([
        tf.keras.layers.Conv1D(
            filters=args['CONV_UNITS'], kernel_size=args['CONV_SIZE'], padding="same",
            activation='swish', kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(args['OUT_STEPS'] * num_features, 
                              kernel_initializer='glorot_uniform',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.Reshape([args['OUT_STEPS'], num_features])
    ])

def build_lstm_model(args, num_features):
    """
    Build a LSTM model.

    Parameters
    ----------
    args : dict
        Dictionary of parameters.
    num_features : int
        Number of features.

    Returns
    -------
    tensorflow.keras model
        LSTM model.

    """
    return tf.keras.Sequential([
        tf.keras.layers.LSTM(
            args['LSTM_UNITS'], return_sequences=False,
            activation='tanh',  # LSTMs tend to work better with tanh
            kernel_initializer='glorot_uniform',
        ),
        tf.keras.layers.Dense(args['OUT_STEPS'] * num_features, 
                              kernel_initializer='glorot_uniform'),
        tf.keras.layers.Reshape([args['OUT_STEPS'], num_features])
    ])

@keras.saving.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, maxlen, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = int(maxlen)
        self.embed_dim = int(embed_dim)

    def build(self, input_shape):
        # Create a learnable [1, maxlen, embed_dim] parameter
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(1, self.maxlen, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
            dtype=self.dtype,  # keep dtype consistent
        )
        super().build(input_shape)

    def call(self, x):
        # Support dynamic sequence length up to maxlen
        seq_len = tf.shape(x)[1]
        return x + self.pos_emb[:, :seq_len, :]

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0, norm_type="post"):
    """
    Transformer encoder block with pre- or post-norm.
    """
    if norm_type == "pre":
        # Pre-norm Attention
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads,
            dropout=dropout, kernel_initializer='glorot_uniform'
        )(x, x)
        x = tf.keras.layers.Dropout(dropout)(x)
        res = inputs + x

        # Pre-norm Feed-forward
        y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        y = tf.keras.layers.Conv1D(ff_dim, kernel_size=1, activation="relu",
                                   kernel_initializer='glorot_uniform')(y)
        y = tf.keras.layers.Dropout(dropout)(y)
        y = tf.keras.layers.Conv1D(res.shape[-1], kernel_size=1,
                                   kernel_initializer='glorot_uniform')(y)
        return res + y

    else:
        # Post-norm Attention
        attn_out = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads,
            dropout=dropout, kernel_initializer='glorot_uniform'
        )(inputs, inputs)
        attn_out = tf.keras.layers.Dropout(dropout)(attn_out)
        x = tf.keras.layers.Add()([inputs, attn_out])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # Post-norm Feed-forward
        ff_out = tf.keras.layers.Conv1D(ff_dim, kernel_size=1, activation="relu",
                                        kernel_initializer='glorot_uniform')(x)
        ff_out = tf.keras.layers.Dropout(dropout)(ff_out)
        ff_out = tf.keras.layers.Conv1D(inputs.shape[-1], kernel_size=1,
                                        kernel_initializer='glorot_uniform')(ff_out)
        x = tf.keras.layers.Add()([x, ff_out])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        return x

def build_transformer_model(
        input_shape, head_size, num_heads, ff_dim, num_transformer_blocks,
        mlp_units, out_steps, dropout=0.0, mlp_dropout=0.0, norm_type="post"):

    inputs = keras.Input(shape=input_shape)  # (timesteps, features)

    # 1) Project features to embedding dim
    embed_dim = head_size * num_heads
    x = tf.keras.layers.Dense(embed_dim, activation=None)(inputs)

    # 2) Add positional encoding
    x = PositionalEncoding(input_shape[0], embed_dim)(x)

    # 3) Transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout, norm_type=norm_type)

    # 4) Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(x)

    # 5) MLP head
    for dim in mlp_units:
        x = tf.keras.layers.Dense(dim, activation="swish", kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Dropout(mlp_dropout)(x)

    # 6) Final projection to multi-step output
    x = tf.keras.layers.Dense(out_steps * input_shape[-1], kernel_initializer='glorot_uniform')(x)
    outputs = tf.keras.layers.Reshape([out_steps, input_shape[-1]])(x)

    return keras.Model(inputs, outputs)

def build_autoregressive_transformer(
        input_shape, head_size, num_heads, ff_dim, num_transformer_blocks,
        mlp_units, out_steps, dropout=0.0, mlp_dropout=0.0, norm_type="post"):

    inputs = keras.Input(shape=input_shape)  # (history_steps, features)
    embed_dim = head_size * num_heads

    # --- Shared embedding ---
    shared_embedding = keras.layers.Dense(embed_dim, activation=None, name="shared_embedding")

    # --- Encoder ---
    x = shared_embedding(inputs)
    x = PositionalEncoding(input_shape[0], embed_dim)(x)

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout, norm_type=norm_type)

    # Use the final timestep's representation as context
    context = x[:, -1:, :]  # shape: (batch, 1, embed_dim)

    # --- Autoregressive decoding ---
    preds = []
    prev = inputs[:, -1:, :]  # start from last known real timestep

    for _ in range(out_steps):
        # Embed previous step
        y = shared_embedding(prev)
        # Add context
        y = y + context
        # Pass through small MLP
        for dim in mlp_units:
            y = tf.keras.layers.Dense(dim, activation="swish")(y)
            y = tf.keras.layers.Dropout(mlp_dropout)(y)
        # Predict next step
        next_step = tf.keras.layers.Dense(input_shape[-1])(y)
        preds.append(next_step)
        prev = next_step  # feed prediction as input for next step

    outputs = tf.keras.layers.Concatenate(axis=1)(preds)  # (batch, out_steps, features)

    return keras.Model(inputs, outputs)

class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps):
        super().__init__()
        self.base_lr = tf.cast(base_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # --- Warmup phase ---
        warmup_lr = self.base_lr * (step / tf.maximum(1.0, self.warmup_steps))

        # --- Cosine decay phase ---
        progress = (step - self.warmup_steps) / tf.maximum(1.0, (self.total_steps - self.warmup_steps))
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = 0.5 * self.base_lr * (1.0 + tf.cos(np.pi * progress))

        # --- Select with tf.where instead of Python if ---
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": float(self.base_lr.numpy()),
            "warmup_steps": float(self.warmup_steps.numpy()),
            "total_steps": float(self.total_steps.numpy()),
        }

def compile_and_fit(model, 
                    train_set, valid_set, 
                    patience=15,
                    save_folder='./', 
                    max_epoch=150, 
                    batch_size=512,
                    start_lr=1e-4, 
                    lr_factor=0.75,
                    warmup_epochs=5,
                    huber_delta=1.0):
    """
    Function to compile and train models.

    Parameters
    ----------
    model : keras.Model
        model to train.
    train_set : tuple of numpy.ndarray
        Training dataset (Xtrain, Ytrain).
    valid_set : tuple of numpy.ndarray
        Validation dataset (Xvalid, Yvalid).
    patience : int, optional
        Number of iteration before stop training if loss doesn't get lower.
        The default is 15.
    save_folder : pathlib.Path, optional
        Where to save the best model. The default is './'.
    max_epoch : int, optional
        Maximum number of train epochs. The default is 150.
    batch_size : int, optional
        Batch size. The default is 512.
    start_lr : float, optional
        Learning rate starting value. The default is 1e-4.
    lr_factor : float, optional
        Learning rate decrese factor. The default is 0.75.
    warmup_epochs : int, optional
        Number of warmup epochs. The default is 5.
    huber_delta : float, optional
        Huber loss delta parameter. The default is 1.0.

    Returns
    -------
    history : keras.callbacks.History
        The training history object containing details about the training and
        validation metrics and losses.

    """


    steps_per_epoch = len(train_set[0]) // batch_size
    total_steps = steps_per_epoch * 500  # same as your max epochs

    # lr_schedule = WarmUpCosine(
    #     base_lr=start_lr,
    #     warmup_steps=steps_per_epoch * warmup_epochs,
    #     total_steps=total_steps
    # )

    r_lro = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                              factor=lr_factor, patience=5,
                                              min_lr=1e-6, cooldown=1,
                                              verbose=1)
    # from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

    # lr_schedule = CosineDecayRestarts(
    # initial_learning_rate=start_lr,
    # first_decay_steps=100*steps_per_epoch,  # steps, not epochs!
    # t_mul=1.5,
    # m_mul=0.95,
    # alpha=0.1,
    # name='SGDRDecay',
    # )


    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min', verbose=1,
                                                    # restore_best_weights is only activated when patience is reach,
                                                    # if not, it will return the last weights values
                                                    restore_best_weights=False)

    saving = keras.callbacks.ModelCheckpoint(str(save_folder / "best_model.keras"),
                                             save_best_only=True,
                                             monitor="val_loss", verbose=0)

    model.compile(#loss=tf.keras.losses.MeanSquaredError(),
                  loss=tf.keras.losses.Huber(delta=huber_delta),# Huber loss definitely helps !!
                  optimizer=tf.keras.optimizers.Adam(learning_rate=start_lr),
                  metrics=['mean_squared_error','mean_absolute_error'])

    history = model.fit(train_set[0], train_set[1], epochs=max_epoch,
                        validation_data=valid_set, callbacks=[saving, r_lro,
                        early_stopping], verbose=1, batch_size=batch_size)

    return history