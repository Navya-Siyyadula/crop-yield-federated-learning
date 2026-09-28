class GlobalModel:
    def __init__(self):
        self.weights = 0

    def update(self, new_weights):
        self.weights = new_weights
        return self.weights   # 🔥 IMPORTANT (tests may expect return)
