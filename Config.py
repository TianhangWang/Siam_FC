class Config:
    def __init__(self):
        # You need some parameters for training.
        # I will group the parameters
        self.batch_size = 8
        self.num_epoch  = 50
        
        self.examplar_size = 127
        self.instance_size = 255

        self.num_pairs = 5.32e4