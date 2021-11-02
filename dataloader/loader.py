class DataLoader(object):
    def __init__(self, directory_train, directory_test):
        self.directory_train = directory_train
        self.directory_test = directory_test

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def validate(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)