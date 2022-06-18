import dataloading
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def main():
    sentences_train = dataloading.load_raw_dataset('train')
    sentences_eval = dataloading.load_raw_dataset('eval')

    feature_maps = [lambda sentence: [word[0].isupper() for word in sentence[0]],
                    lambda sentence: [True if ii == 0 else False for ii in range(len(sentence[0]))],
                    lambda sentence: [True if pos == 'NNP' else False for pos in sentence[1]],
                    lambda sentence: [True if pos in ('NN', 'CD', 'IN', 'DT') else False for pos in sentence[1]],
                    lambda sentence: [True if last_pos == 'NNP' else False for last_pos in sentence[2]],
                    lambda sentence: [True if next_pos == 'NNP' else False for next_pos in sentence[3]],
                    lambda sentence: [len(word) > 3 for word in sentence[0]]
                    ]
    X_train, y_train = dataloading.convert_raw_to_features(sentences_train, feature_maps)
    X_eval, y_eval = dataloading.convert_raw_to_features(sentences_eval, feature_maps)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    score = model.score(X_eval, y_eval)
    print(score)
    print(model.feature_importances_)
    return


if __name__ == '__main__':
    main()