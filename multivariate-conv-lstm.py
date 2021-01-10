'''
Script to make a Multivariate Encoder Decoder LSTM Model
'''
import math
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot
from random import shuffle
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import model_from_json
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from keras.layers import ConvLSTM2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


PATH_TO_TEST_TRAIN_DATA = 'data/segregatedData-Weekly/train-test/'
PATH_TO_SAVE_RESULT = 'results/exp4/'
if not os.path.exists(PATH_TO_SAVE_RESULT):
    os.makedirs(PATH_TO_SAVE_RESULT)
    os.makedirs(PATH_TO_SAVE_RESULT + 'Error_Plots/')

#UNISON Shuffle
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# convert our Regression Problem into supervised Learning
def to_supervised(train, n_input, n_out):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    x, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
            x.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return np.array(x), np.array(y)


def build_skeleton_model(n_timesteps, n_features, n_outputs):
    # Define skeleton model
    model = Sequential()
    '''ENCODER-DECODER-LSTM'''
    # model.add(LSTM(400, activation='relu', input_shape=(n_timesteps, n_features)))
    # model.add(RepeatVector(n_outputs))
    # model.add(LSTM(200, activation='relu', return_sequences=True))
    # model.add(TimeDistributed(Dense(100, activation='relu')))
    # model.add(TimeDistributed(Dense(1)))


    '''VANILA LSTM'''
    # model.add(LSTM(400, return_sequences=False, input_shape=(n_timesteps, n_features)))
    # model.add(Dense(1))

    '''STACKED LSTM'''
    # model.add(LSTM(400, return_sequences=True, input_shape=(n_timesteps, n_features)))
    # model.add(LSTM(100))
    # model.add(Dense(1))

    '''BIDIRECTIONAL LSTM'''
    # model.add(Bidirectional(LSTM(400, input_shape=(n_timesteps, n_features))))
    # model.add(Dense(1))

    '''CNN-LSTM n_timesteps'''
    # model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_features, n_outputs)))
    # model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    # model.add(TimeDistributed(Flatten()))
    # model.add(LSTM(50, activation='relu'))
    # # model.add(LSTM(200, activation='relu',  return_sequences=True))
    # # model.add(LSTM(100))
    # model.add(Dense(1))

    '''CONV-LSTM '''
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu', input_shape=(None, 1, n_features, n_outputs)))
    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model

def build_model(train, number_of_lags, number_of_prediction_step, take_old_model = True):
    # check a Preexisting Model
    model_file_folder = PATH_TO_SAVE_RESULT + 'Model/'
    if os.path.isfile(model_file_folder + 'model.json') and os.path.isfile(model_file_folder + 'model.h5' and take_old_model):
        json_file = open(model_file_folder + 'model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(model_file_folder + "model.h5")
        return loaded_model
    # OW If No Previous Model Exist Create A Dir
    if not os.path.exists(PATH_TO_SAVE_RESULT + 'Model/'):
        os.makedirs(PATH_TO_SAVE_RESULT + 'Model/')
    # Start the Training Process
    # print("Deeper Insight", train['1-3'])
    # exit()
    train_dict_x, train_dict_y = dict(), dict()
    for sensor_id in train:
        train_dict_x[sensor_id], train_dict_y[sensor_id] = to_supervised(train[sensor_id], number_of_lags,
                                                                         number_of_prediction_step)
    # Get Any ID
    tmp_id = next(iter(train))
    # Merge all X Y Pairs and Randomize it
    # train_x, train_y = list(), list()
    train_x, train_y = train_dict_x[tmp_id], train_dict_y[tmp_id]
    for sensor_id in train_dict_x:
        if sensor_id != tmp_id:
            train_x = np.concatenate((train_x, train_dict_x[sensor_id]))
            train_y = np.concatenate((train_y, train_dict_y[sensor_id]))
    # Randomize it
    train_x, train_y = unison_shuffled_copies(train_x, train_y)

    # define parameters
    verbose, epochs, batch_size = 0, 15, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]


    # print("\nInsight in training\n", train_x[5], "\nTest\n", train_y[5], np.shape(train_x))
    # exit()

    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1,train_x.shape[2], 1))
    print("train_x", train_x.shape)
    # exit()
    #DEBUG
    #print("DEBUG", train_x.shape[1], train_x.shape[2], train_x.shape)
    # print("Insight in training ", train_x[5], "Test", train_y)

    # Get Skeleton Model
    model = build_skeleton_model(n_timesteps, n_features, n_outputs)
    # reshape output into [samples, timesteps, features]
    # if not (n_outputs == 1):
    #     train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], n_outputs))
    # DEBUG
    train_y = train_y.reshape((train_y.shape[0]))
    # print("DEBUG-2", train_y.shape)
    # # print(train_y)
    # exit()
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # save network
    model_json = model.to_json()
    with open(model_file_folder + "model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_file_folder + "model.h5")
    return model


# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = np.array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # print("data-shape",data.shape)
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], 1,input_x.shape[1], 1))
    # print("input", input_x.shape)
    # exit()
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat

def make_me_graphs(actual, predicted, name):
    # print("\nactual.shape[0]:", actual.shape[0])
    ts = np.linspace(start = 1, stop = actual.shape[0], num = actual.shape[0])
    # print("\n ts", ts)
    ts = ts.reshape(ts.shape[0], 1)
    # print("\n ts.shape", ts.shape)
    # exit()
    # print("\nActual shape: ", actual.shape)
    # print("\n Predicted shape", predicted.shape)
    pyplot.figure()
    pyplot.plot(ts, actual)
    pyplot.plot(ts, predicted)
    pyplot.xlabel('Time')
    pyplot.ylabel('Data-plot')
    pyplot.title('CONV-LSTM')
    pyplot.savefig('plots/' + name + '.png')

def make_my_graph2(actual_train, train_predicted, actual_test, test_predicted, name):

    '''Convert to Degrees'''
    actual_test = actual_test * 0.05729
    actual_train = actual_train * 0.05729
    train_predicted = train_predicted * 0.05729
    test_predicted = test_predicted * 0.05729

    ts_train = np.linspace(start=1, stop=actual_train.shape[0], num=actual_train.shape[0])
    ts_train = ts_train.reshape(ts_train.shape[0], 1)

    ts_test = np.linspace(start= actual_train.shape[0] + 1, stop=actual_train.shape[0] + actual_test.shape[0], num=actual_test.shape[0])
    ts_test = ts_test.reshape(ts_test.shape[0], 1)

    '''Finding Min Max Scale for Soil Movements'''
    max_train, min_train = np.amax(actual_train), np.amin(actual_train)
    max_test, min_test = np.amax(actual_test), np.amin(actual_test)

    true_max = max(max_train, max_test, min_train, min_test)
    true_min = min(max_train, max_test, min_train, min_test)

    soil_movement_horizontal = np.linspace(start=true_min-1, stop=true_max+1)
    mid_point_for_train_and_test = (actual_train.shape[0] + actual_train.shape[0] + 1)/2
    ts_horizontal = [mid_point_for_train_and_test for x in range(len(soil_movement_horizontal))]

    '''Quick Debug'''
    print(max_train, max_test, min_train, min_test)

    '''Getting new instance of pyplot'''
    pyplot.figure()

    '''Plot the Training Actual and Training Learned'''
    pyplot.plot(ts_train, actual_train, label='Training Actual')
    pyplot.plot(ts_train, train_predicted, label='Training Learned')

    '''Plot the Testing Actual and Testing Predicted'''
    pyplot.plot(ts_test, actual_test, label='Testing Actual')
    pyplot.plot(ts_test, test_predicted, label='Testing Predicted')

    '''Plotting Horizontal line for better visualization'''
    pyplot.plot(ts_horizontal, soil_movement_horizontal, color='black', linestyle='dashed')

    '''Putting Labels and Legends on Plot'''
    pyplot.xlabel("Weeks")
    pyplot.ylabel("Soil Movements")

    pyplot.legend(loc='upper left')


    pyplot.title("CONV-LSTM")

    '''Save our figure'''
    pyplot.savefig('plots/CONV_LSTM/' + name + '.png')


def evaluate_forecasts(actual, predicted, name):

    # make_me_graphs(actual, predicted, name)

    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        # print("shape of actual ", np.shape(actual[:, i]), "shape of predicted", np.shape(predicted[:, i]))
        # tmp = actual[:, i]
        # tmp1 = predicted[:, i]
        # print("Type of tmp", type(tmp))
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # exit()
        # calculate rmse
        rmse = math.sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = math.sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


def restructure_data(train_dict, test_dict, number_of_prediction_step):
    for sensor_id in train_dict:
        # Remove Values which cannot form windows
        train_dict[sensor_id] = train_dict[sensor_id].values[:(int(len(train_dict[sensor_id])/number_of_prediction_step)
                                                               ) * number_of_prediction_step]
        test_dict[sensor_id] = test_dict[sensor_id].values[:(int(len(test_dict[sensor_id])/number_of_prediction_step)
                                                             ) * number_of_prediction_step]
        train_dict[sensor_id] = np.array(np.split(train_dict[sensor_id],
                                                  len(train_dict[sensor_id])/number_of_prediction_step))
        test_dict[sensor_id] = np.array(np.split(test_dict[sensor_id],
                                                 len(test_dict[sensor_id])/number_of_prediction_step))
    return train_dict, test_dict


def evaluate_model(train_dict, test_dict, number_of_lags, number_of_prediction_step):
    # print("Shape of Vectors before restructuring", np.shape(train_dict['1-3']), np.shape(test_dict['1-3']))
    train_dict, test_dict = restructure_data(train_dict, test_dict, number_of_prediction_step)
    # print("The Big Bang train_dict", train_dict['1-3'], "Test", test_dict['1-3'])
    # exit()
    model = build_model(train_dict, number_of_lags, number_of_prediction_step, False)
    # History(Input data) to Predict values
    # It must be dictionary filled with first "number_of_lags" value
    history = dict()
    # print("Shape of Vectors after restructuring", np.shape(train_dict['1-3']), np.shape(test_dict['1-3']))
    # print(np.shape(test_dict['1-3']))
    # for sensor_id in test_dict:
    #     print(sensor_id)
    #     print(np.shape(test_dict[sensor_id][:number_of_lags]))
    # exit()
    for sensor_id in test_dict:
        history[sensor_id] = [x for x in test_dict[sensor_id][:number_of_lags]]
    # walk-forward validation over each week
    predictions = dict()
    for sensor_id in test_dict:
        test = test_dict[sensor_id]
        predictions[sensor_id] = list()
        # print("shape of test in line 167", np.shape(test))
        for i in range(len(test) - number_of_lags - number_of_prediction_step):
            # predict the week
            # print("shape of parameter of forecast", np.shape(history[sensor_id]))
            yhat_sequence = forecast(model, history[sensor_id], number_of_lags)
            # store the predictions
            # print("yhat_sequence", np.shape(yhat_sequence))
            predictions[sensor_id].append(yhat_sequence)
            # print("prediction after append", np.shape(predictions[sensor_id]))
            # get real observation and add to history for predicting the next week
            history[sensor_id].append(test[i+number_of_lags, :])
            # print("history after append", np.shape(history[sensor_id]))
            # exit()

    # For Training Accuracy
    history_train = dict()
    for sensor_id in train_dict:
        history_train[sensor_id] = [x for x in train_dict[sensor_id][:number_of_lags]]

    # walk-forward validation over each week
    predictions_train = dict()
    for sensor_id in train_dict:
        test = train_dict[sensor_id]
        predictions_train[sensor_id] = list()
        for i in range(len(test) - number_of_lags - number_of_prediction_step):
            yhat_sequence = forecast(model, history_train[sensor_id], number_of_lags)
            predictions_train[sensor_id].append(yhat_sequence)
            history_train[sensor_id].append(test[i+number_of_lags, :])

    # evaluate predictions
    score = dict()
    scores = dict()

    # Getting Training Accuracy
    score_train, scores_train = dict(), dict()

    for sensor_id in test_dict:
        # print("Parameters of evaluate_forecast", np.shape(test_dict[sensor_id][number_of_lags:len(test_dict[sensor_id]) - number_of_prediction_step][:, :, 0]) , np.shape(np.array(predictions[sensor_id])))
        # exit()

        test_data = test_dict[sensor_id][number_of_lags:len(test_dict[sensor_id]) - number_of_prediction_step][:, :, 0]
        prediction_test = np.array(predictions[sensor_id])

        score_, scores_ = evaluate_forecasts(test_data, prediction_test, "Test "+str(sensor_id))

        train_data = train_dict[sensor_id][number_of_lags:len(train_dict[sensor_id]) - number_of_prediction_step][:, :, 0]
        prediction_train = np.array(predictions_train[sensor_id])

        score_train_, scores_train_ = evaluate_forecasts(train_data, prediction_train, "Train " + str(sensor_id))

        score[sensor_id] = score_
        scores[sensor_id] = scores_

        score_train[sensor_id] = score_train_
        scores_train[sensor_id] = scores_train_

        '''Plot Curves'''
        make_my_graph2(train_data, prediction_train, test_data, prediction_test, str(sensor_id))

    return score, scores, score_train, scores_train


# summarize scores
def summarize_scores(name, score, scores):
    avg_rmse = 0
    s = ""
    for sensor_id in score:
        s_scores = ' & '.join(['%.3f' % s for s in scores[sensor_id]])
        #print('%s: %s [%.3f] %s' % (name, str(sensor_id), score[sensor_id], s_scores))
        avg_rmse += 0.05729*score[sensor_id]
        s += str("{0:.4f}".format(0.05729 * score[sensor_id])) + " & "

    s += str("{0:.4f}".format(avg_rmse / 5.0))
    # Print the average RMSE Score
    #print("Average RMSE score", avg_rmse/5.0)
    print("STRING", s)


def main():
    # Creating 2 dictionary to Hold Test and Train Data respectively
    train, test = dict(), dict()
    # Iterating over data to create 2-D matrix of Train and Test
    for boreHole, boreDepth in [(1, 3), (2, 12), (3, 6), (4, 15), (5, 15)]:
        sensor_id = str(boreHole) + '-' + str(boreDepth)
        train[sensor_id] = pd.read_csv(PATH_TO_TEST_TRAIN_DATA + 'Train/' + sensor_id + '.csv',
                                        header=0, usecols=[0, 1, 2, 3], index_col=0)
        test[sensor_id] = pd.read_csv(PATH_TO_TEST_TRAIN_DATA + 'Test/' + sensor_id + '.csv',
                                        header=0, usecols=[0, 1, 2, 3], index_col=0)
            # print(list(train[sensor_id].columns.values))
            # exit()
            #train[sensor_id] = train[sensor_id][['Bore Hole', 'Meter', 'Data plot ']]
            #test[sensor_id] = test[sensor_id][['Bore Hole', 'Meter', 'Data plot ']]
    # "Number of Past values" to consider to predict "Number of Future Values" all at a frequency
    # of the data set i.e 1 hour in Tangri Data-Set
    # print("In the end", train['1-3'], "Test",test['1-3'])
    number_of_lags = 2
    number_of_prediction_step = 1
    score, scores, score_train, scores_train = evaluate_model(train, test, number_of_lags, number_of_prediction_step)
    # Print Result
    summarize_scores('LSTM-VANILA-WITH-BATCH-TESTING', score, scores)
    summarize_scores('LSTM-VANILA-WITH-BATCH-TRAINING', score_train, scores_train)

    # plot scores & Save
    prediction_labels = [str(x+1) + 'hr' for x in range(number_of_prediction_step)]
    for sensor_id in scores:
        pyplot.figure()
        pyplot.plot(prediction_labels, scores[sensor_id], marker='o', label='lstm')
        pyplot.savefig(PATH_TO_SAVE_RESULT + 'Error_Plots/' + str(sensor_id) + '.png')
        pyplot.close()


if __name__ == '__main__':
    main()
