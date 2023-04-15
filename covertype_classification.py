import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, cohen_kappa_score, matthews_corrcoef, confusion_matrix

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Embedding, Dense, BatchNormalization, Dropout, Concatenate, ReLU
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

from lr_cbs import LRFinder

# SEED FOR REPLICABILITY
np.random.seed(42)
tf.random.set_seed(42)

# DEFINE PATHS
DATASET_PATH = 'dataset/covtype.data'
CHART_SAVEDIR = 'charts/'
MODEL_SAVEDIR = 'models/'

# DEFINE DATASET CONSTANTS
NUM_CLASSES = 7         # the number of target classes
CONT_SIZE = 10          # the number of continous variables
CAT_SIZE = [4, 40]      # the number of categories in a class variable

NAMES_CONT   = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
NAMES_CAT    = [f'Wilderness_Area{i+1}' for i in range(4)] + [f'Soil_Type{i+1}' for i in range(40)]
NAMES_TARGET = ['Cover_Type']


def xy_split(df):
    """
    Splits DataFrame into features and labels.

    Arguments:
        df {pandas.DataFrame} -- A pandas DataFrame containing the input features and target values.

    Returns:
        tuple -- A tuple of two numpy arrays: features, labels.
                 The features are ordered according to the dataset description.
    """
    return df[NAMES_CONT+NAMES_CAT].to_numpy().astype(np.float64), df[NAMES_TARGET].to_numpy().flatten()


def simple_heuristic(X, proba=False):
    """
    A simple OneR model that predicts one of two dominant classes based on Elevation.

    Arguments:
        X : np.ndarray -- Input feature matrix of shape (n_samples, n_features) before scaling.
        proba : bool, default=False -- Whether to return predicted class probabilities.

    Returns:
        np.ndarray -- If `proba` is False, returns predicted class labels of shape (n_samples,).
                      If `proba` is True, returns predicted class probabilities of shape (n_samples, NUM_CLASSES).
    """
    predictor = X[:, 0] > 3044.5
    if not proba:
        return np.where(predictor, 0, 1)
    
    proba = np.zeros((X.shape[0], NUM_CLASSES))
    proba[:, 0] = predictor
    proba[:, 1] = 1 - predictor
    
    return proba


def get_class_weights(y_train):
    """
    Computes class weights for imbalanced classification problems suitable for tf.keras.Model.fit.

    Arguments:
        y_train {numpy.ndarray} -- Array of true class labels of shape (n_samples,).

    Returns:
        dict -- A dictionary mapping each class label to its corresponding weight.
    """
    class_weights = compute_class_weight('balanced',
                                        classes=np.unique(y_train),
                                        y=y_train)
    return {it: w for it, w in enumerate(class_weights)}


def get_callbacks():
    """
    Returns a list of callbacks for tf.keras.Model.fit.

    Returns:
        list of tf.keras.callbacks.Callback -- A list of callbacks.
    """
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5
    )
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=20,
        restore_best_weights=True
    )
    return [reduce_lr, early_stop]


def create_model(emb_size, hidden_size, dropout_rate, opt=None):
    """
    Creates a neural network with both continuous and categorical features as inputs using Keras.
    Categorical embeddings and continous variables into a series of Dense -> ReLU -> BatchNorm -> Dropout.
    
    Arguments:
        emb_size {list of int} -- List of integers representing the size of embeddings for each categorical feature.
        hidden_size {list of int} -- List of integ ers representing the size of each hidden layer.
        dropout_rate {float} -- The dropout rate.
        opt {tf.keras.optimizers.Optimizer | string} -- Name of optimizer or optimizer instance/

    Returns:
        A compiled Keras model.
    """
    assert len(emb_size) == len(CAT_SIZE)
        
    input_size = CONT_SIZE + sum(CAT_SIZE)
    input_layer = Input(shape=(input_size,), name='input')
    
    x_cont = BatchNormalization()(input_layer[:, :CONT_SIZE])

    cat_size = [0] + CAT_SIZE
    x_cat = Concatenate()([Embedding(input_dim=cs, output_dim=es)(tf.argmax(input_layer[:, CONT_SIZE+cs_m1:CONT_SIZE+cs], axis=1)) for cs_m1, cs, es in zip(cat_size[:-1], cat_size[1:], emb_size)])
    x = Concatenate()([x_cont, x_cat])
    
    for it, h_size in enumerate(hidden_size):
        x = Sequential([
            Dense(h_size, use_bias=False),
            ReLU(),
            BatchNormalization(),
            Dropout(rate=dropout_rate)
        ], name=f'LinBnDrop_{it}')(x)

    x = Dense(NUM_CLASSES, activation='softmax', name='output')(x)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=x)
    model.compile(
                optimizer='adam' if not opt else opt,
                loss=SparseCategoricalCrossentropy(),
                metrics=['accuracy']
            )
    return model


def plot_training_curves(history):
    """
    Plots the loss and accuracy curves of a given model's history.

    Args:
        history (keras.callbacks.History): The history object of the trained model.

    Returns:
        matplotlib.pyplot.figure: A figure containing two horizontal subplots, one for loss and the other for accuracy.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Training curves', fontsize='x-large')

    axs[0].plot(history.history['loss'], label='Training Loss')
    axs[0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(history.history['accuracy'], label='Training Accuracy')
    axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()

    return fig


def combine_figs(*figs):
    """
    Combines two or more matplotlib figures into a single figure by staking them vertically.

    Arguments:
        *figs {matplotlib.figure.Figure} -- Two or more matplotlib figure objects.

    Returns:
        matplotlib.figure.Figure -- A new matplotlib figure object that combines the input figures.
    """
    assert len(figs)
    if len(figs) == 1:
        return figs[0]
    
    fig1, fig2 = figs[0], figs[1]
    fig3 = plt.figure()

    size1 = fig1.get_size_inches()
    size2 = fig2.get_size_inches()

    total_width = max(size1[0], size2[0])
    total_height = size1[1] + size2[1]

    offset1 = (total_width - size1[0]) / 2.0
    offset2 = (total_width - size2[0]) / 2.0

    renderer = fig1.canvas.get_renderer()
    fig1.draw(renderer)
    fig3.figimage(fig1.canvas.buffer_rgba(),  xo=offset1*fig1.dpi, yo=(total_height-size1[1])*fig1.dpi)

    renderer = fig2.canvas.get_renderer()
    fig2.draw(renderer)
    fig3.figimage(fig2.canvas.buffer_rgba(), xo=offset2*fig2.dpi, yo=0)

    fig3.set_size_inches(total_width, total_height)
    
    plt.close(fig1)
    plt.close(fig2)

    return combine_figs(fig3, *figs[2:])


def metric_generator(y_true, *y_pred):
    """
    A generator that calculates desired metrics.

    Arguments:
        y_true {numpy.ndarray} -- The true labels of shape (n_samples,).
        y_pred {numpy.ndarray} -- A variable number of predicted probability arrays or label arrays of shape (n_samples, n_classes).
                                  Can provide multiple arrays to compare the performance of different models.

    Yields:
        tuple of (string, list of float) -- the name of the metric with a list of the metric's scores for each of the probability arrays
    """
    def w(metric, proba=False, **kwargs):
        """
        Wrapper around metrics function.

        Arguments:
            metric {callable} -- A scoring function from scikit-learn.metrics to apply.
            proba {boolean} -- Whether the predicted values are probabilities (True) or labels (False).
            **kwargs -- Additional keyword arguments to pass to the metric function.
        """
        return metric.__name__, [metric(y_true, y_pr if proba else np.argmax(y_pr, axis=1), **kwargs) for y_pr in y_pred]
        
    # yield w(accuracy_score)
    yield w(f1_score,           average='weighted')
    yield w(log_loss,           proba=True)
    yield w(roc_auc_score,      proba=True, multi_class='ovr', average='weighted')
    yield w(cohen_kappa_score)
    yield w(matthews_corrcoef)


def plot_metrics(y_true, y_pred):
    """
    Generates a set of plots that show the performance of each model across multiple evaluation metrics.

    Arguments:
        y_true {numpy.ndarray} -- The true labels of shape (n_samples,).
        y_pred {dict of (string, numpy.ndarray)} -- A dictionary of the names of the model and predicted label probability arrays of shape (n_samples, n_classes),

    Returns:
        matplotlib.pyplot.figure -- A figure object that contains the generated plots.
    """
    num_metrics = sum(1 for _ in metric_generator(y_true, *y_pred.values()))

    fig, axs = plt.subplots(nrows=1, ncols=num_metrics, figsize=(4*num_metrics, 4), dpi=300)
    fig.suptitle('Evaluation of models across different metrics', fontsize='x-large')

    for it, (metric_name, scores) in enumerate(metric_generator(y_true, *y_pred.values())):
        axs[it].bar(y_pred.keys(), scores, width=0.5, zorder=3)
        axs[it].set_title(metric_name)
        axs[it].set_ylim(0.9*min(scores), 1.1*max(scores))
        axs[it].grid(zorder=0)
        plt.setp(axs[it].get_xticklabels(), rotation=45, ha='right')

        if metric_name == 'log_loss':
            ylim = max(scores[1:])
            axs[it].annotate(f"{scores[0]:.2f}",
                xy=(0, 1.1*ylim),
                xytext=(0, 1.15*ylim),
                ha='center', va='center', 
                fontsize=10,
                arrowprops=dict(arrowstyle='->', linewidth=0.5))
            axs[it].set_ylim(0.9*min(scores), 1.1*ylim)
    
    plt.subplots_adjust(bottom=0.3)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(y_true, y_pred):
    """
    Generates a set of plots that represent confusion matrices of models.

    Arguments:
        y_true {numpy.ndarray} -- The true labels of shape (n_samples,).
        y_pred {dict of (string, numpy.ndarray)} -- A dictionary of predicted probability arrays or label arrays of shape (n_samples, n_classes),
                        keyed by the names of the models.
    Returns:
        matplotlib.pyplot.figure -- a figure of confusion matrix plots
    """
    cfms = [confusion_matrix(y_true, np.argmax(y_pr, axis=1), normalize='true') for y_pr in y_pred.values()]

    ncols = len(cfms)
    fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols + 1, 4.5), dpi=300)
    fig.suptitle('Confusion matrices; normalized to actual labels', fontsize='x-large')

    for i, (cf, ax, model_name) in enumerate(zip(cfms, axs, y_pred.keys())): 
        sns.heatmap(cf, annot=True, cmap='Blues', ax=ax, fmt='.2f', cbar=i==len(axs)-1, annot_kws={"size": 9})
        ax.set_title(model_name)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
    fig.tight_layout()
    return fig


def main():
    # LOAD DATA
    df = pd.read_csv(DATASET_PATH, header=None, names=NAMES_CONT+NAMES_CAT+NAMES_TARGET)
    df['Cover_Type'] -= 1 # for class labels to be in range [0, 6]

    # SPLIT INTO TRAIN / VAL
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
    X_train, y_train = xy_split(df_train)
    X_val,   y_val   = xy_split(df_val)

    y_pred = {} # dict for predicted probabilities

    # SIMPLE HEURISTIC
    print('In progress: SimpleHeuristic')
    y_pred['SimpleHeuristic'] = simple_heuristic(X_val, proba=True)

    # NORMALIZE CONTINOUS VARIABLES
    scaler = StandardScaler()
    scaler.fit(X_train[:, :CONT_SIZE])

    X_train[:, :CONT_SIZE] = scaler.transform(X_train[:, :CONT_SIZE])
    X_val[:, :CONT_SIZE]   = scaler.transform(X_val[:, :CONT_SIZE])

    # KNN
    print('In progress: KNeighborsClassifier')
    neigh = KNeighborsClassifier(n_neighbors=12, n_jobs=-1)
    neigh.fit(X_train, y_train)
    y_pred['KNeighbors'] = neigh.predict_proba(X_val)

    # RANDOM FOREST
    print('In progress: RandomForestClassifier')
    rf = RandomForestClassifier(100, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred['RandomForest'] = rf.predict_proba(X_val)

    # NEURAL NET
    print('In progress: NeuralNetwork')
    BATCH_SIZE = 2048
    EPOCHS = 200

    # FIND OPTIMAL LR
    lr_finder = LRFinder(start_lr=1e-6, end_lr=3)
    nn_lr_find = create_model(
                emb_size=[5, 15], 
                hidden_size=[300, 150, 50], 
                dropout_rate=0.0,
                opt='adam'
            )
    
    nn_lr_find.fit(X_train, y_train, epochs=1, callbacks=[lr_finder], verbose=False)
    fig = lr_finder.plot()
    fig.savefig(CHART_SAVEDIR + 'lr_loss.png')
    
    # TRAIN MODEL
    nn = create_model(
                emb_size=[5, 15],
                hidden_size=[300, 150, 50], 
                dropout_rate=0.0,
                opt=Adam(learning_rate=0.05)
            )

    history = nn.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=get_class_weights(y_train),
        callbacks=get_callbacks()
        )
    
    y_pred['NeuralNetwork'] = nn.predict(X_val)

    fig = plot_training_curves(history)
    fig.savefig(CHART_SAVEDIR + 'training_curves.png', dpi=300)

    # ADD ENSEMBLE - it's for free! but excluding the simplest model
    y_pred['ensemble'] = (sum(y_pred.values()) - y_pred['SimpleHeuristic']) / (len(y_pred) - 1)

    # EVALUATE
    fig = combine_figs(
        plot_metrics(y_val, y_pred),
        plot_confusion_matrices(y_val, y_pred)
    )
    plt.tight_layout()
    fig.savefig(CHART_SAVEDIR + 'evaluation.png', dpi=300)
    # plt.show()

    # SAVE MODELS FOR DEPLOYMENT
    pickle.dump(rf, open(MODEL_SAVEDIR + 'RandomForest.pkl', 'wb'))
    pickle.dump(neigh, open(MODEL_SAVEDIR + 'KNeighbors.pkl', 'wb'))
    
    nn.save(MODEL_SAVEDIR +  'NeuralNet')


if __name__ == '__main__':
    main()