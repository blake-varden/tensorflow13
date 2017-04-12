class Model(object):

    def __init__(self, is_train=False, params=None):
        self.is_train = is_train

    def begin_epoch_ops(self):
        return []

    def begin_epoch_summary(self):
        return None

    def batch_ops(self):
        return []

    def batch_summary(self):
        return None

    def end_epoch_ops(self):
        return []

    def end_epoch_summary(self):
        return None

    def outputs(self):
        pass