from sklearn.svm import SVR

class KernelModel:
    def __init__(self,model_parameters,X_train,Y_train,scaler):
        
        self.M = X_train.shape[1]  # number of items
        self.X_train = X_train  # training set of bundles
        self.Y_train = Y_train  # bidder's values for the bundels in X_train
        self.X_valid = None   # test/validation set of bundles
        self.Y_valid = None  # bidder's values for the bundels in X_valid
        self.model_parameters = model_parameters  # neural network parameters
        self.model = None  # sklearn model
        self.scaler = scaler  # the scaler used fro initially scaling the Y_train values
        self.history = None  # return value of the model.fit() method from keras
        self.loss = None  #TODO return value of the model.fit() method from keras

    def initialize_model(self,regularization_type):
        model = SVR(kernel='linear')
        self.model = model
            
    def fit(self, epochs, batch_size, X_valid=None, Y_valid=None, sample_weight=None):
        # set test set if desired
        self.X_valid = X_valid
        self.Y_valid = Y_valid

        # fit model and validate on test set
        if (self.X_valid is not None) and (self.Y_valid is not None):
            self.history = self.model.fit(X=self.X_train, y=self.Y_train)
        # fit model without validating on test set
        else:
            self.history = self.model.fit(X=self.X_train, y=self.Y_train)
        return 0