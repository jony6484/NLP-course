import dataloading
from sklearn.svm import SVC

def main():
    sentences_train = dataloading.load_raw_dataset('train')
    sentences_eval = dataloading.load_raw_dataset('eval')

    feautre_maps = [lambda sentence: [(ii != 0) and word[0].isupper() for ii, word in enumerate(sentence)],
                   lambda sentence: [len(word) > 8 for word in sentence],
                   lambda sentence: [len(word) < 4 for word in sentence],
                   lambda sentence: ["\',\":@#$%^&*()!" in word for word in sentence]]
    X_train, y_train = dataloading.convert_raw_to_features(sentences_train, feautre_maps)
    X_eval, y_eval = dataloading.convert_raw_to_features(sentences_eval, feautre_maps)

    model = SVC()
    model.fit(X_train, y_train)
    score = model.score(X_eval, y_eval)
    return


if __name__ == '__main__':
    main()