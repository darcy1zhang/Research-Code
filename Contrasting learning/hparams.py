
class hparams():
    def __init__(self):
        self.l2_lambda = 0.02  # L2正则化系数
        self.loss_best = 20
        # self.lr = 0.0001  #tstcc contextual, inital lr for cnn
        self.lr = 0.001  #Pro.song code
        self.batch_size = 16