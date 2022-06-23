import dataloading
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from os import path


def write_pred(file, pred):
    """
    The function creating the prediction file - competitive.txt
    :param file: test file name
    :param pred: our model prediction
    :return: None.
    """
    pred_i = 0
    org_path = path.join('.', 'data', f'{file}.txt')
    pred_path = path.join('.', 'competitive.txt')
    with open(org_path, 'r') as org_file:
        with open(pred_path, 'w') as pred_file:
            for i, line in enumerate(org_file.readlines()):
                if line != '\n':
                    pred_file.write(line.strip() + ' ' + str(pred[pred_i]) + '\n')
                    pred_i += 1
                else:
                    pred_file.write(line)
    return None


def make_feature_maps(which_features):
    """
    A function which defines and reurns the required feature maps
    :param which_features: a list of numbers for choosing the required feature maps
    :return: list of feature maps
    """
    feature_maps = [lambda sentence: [word[0].isupper() for word in sentence[0]],
                    lambda sentence: [ii for ii in range(len(sentence[0]))],
                    lambda sentence: [True if pos in ('NNP', 'NNPS', 'JJS') else False for pos in sentence[1]],
                    lambda sentence: [True if pos in ('NN', 'CD', 'IN', 'DT') else False for pos in sentence[1]],
                    lambda sentence: [True if last_pos == 'NNP' else False for last_pos in sentence[2]],
                    lambda sentence: [True if next_pos == 'NNP' else False for next_pos in sentence[3]],
                    lambda sentence: [len(word) for word in sentence[0]],
                    lambda sentence: [len(sentence[0]) for _ in sentence[0]],
                    lambda sentence: [word[0].lower() in 't,f.o' for word in sentence[0]],
                    lambda sentence: [word[0].lower() in 'jkyzv' for word in sentence[0]]]

    return [feature_maps[ii] for ii in which_features]



def main():
    # Parse the txt data files
    sentences_train = dataloading.load_raw_dataset('train')
    sentences_eval = dataloading.load_raw_dataset('eval')
    sentences_test = dataloading.load_raw_dataset('test', test=True)
    # Define feature maps
    feature_maps = make_feature_maps([0, 2, 3, 6, 7])
    # Make the data for training, evaluating and testing
    X_train, y_train = dataloading.convert_raw_to_features(sentences_train, feature_maps)
    X_eval, y_eval = dataloading.convert_raw_to_features(sentences_eval, feature_maps)
    X_test = dataloading.convert_raw_to_features(sentences_test, feature_maps, test=True)
    # Define the model, fit, score and predict the test
    model = RandomForestClassifier(n_estimators=100, max_depth=13, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_eval, y_eval)
    pred = model.predict(X_test)
    # print(f'model score is {100*score: 0.2f}%')
    # Parse and save the predictions to the "competitive.txt" file
    write_pred('test', pred)
    return


if __name__ == '__main__':
    main()

