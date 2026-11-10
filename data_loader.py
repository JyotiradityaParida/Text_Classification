from datasets import load_dataset

def load_data(n_train=8000, n_test=2000):
    dataset = load_dataset("ag_news")

    train_data = dataset['train'].shuffle(seed=42).select(range(n_train))
    test_data = dataset['test'].shuffle(seed=42).select(range(n_test))

    X_train = train_data['text']
    y_train = train_data['label']
    X_test = test_data['text']
    y_test = test_data['label']

    return X_train, X_test, y_train, y_test