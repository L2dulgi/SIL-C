import cloudpickle

class BasicModule:
    """
    Basic module class using Flax.
    """
    def __init__(self):
        pass

    def create_train_state(self):
        return None
    
    def reinit_optimizer(self):
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError

    def eval_model(self):
        raise NotImplementedError

    def save_model(self, model_path:str, options:dict=None):
        if options['full_saving']:
            with open(model_path, 'wb') as f:
                cloudpickle.dump(self, f)
        else:
            with open(model_path, 'wb') as f:
                cloudpickle.dump(self.train_state.params, f)
        
    def load_model(self, model_path:str, options:dict=None):
        pass
