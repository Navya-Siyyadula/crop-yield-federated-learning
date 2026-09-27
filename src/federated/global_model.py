class GlobalModel:
    def __init__(self):
        self.weights = 0

    def update(self, new_weights):
        print("Updating global model...")
        self.weights = new_weights
