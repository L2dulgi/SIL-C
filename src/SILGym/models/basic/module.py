import cloudpickle

class BasicModule:
    """
    Basic module class using Flax.

    Essential properties that subclasses should define:
    - self.model: Main model instance used for training (required for ModelAppenderV2)
    - self.model_eval: Evaluation model instance (typically with dropout=0.0)
    - self.train_state: Flax TrainState containing params, optimizer state, and apply_fn
    - self.model_config: Dictionary containing model class and configuration kwargs
    - self.optimizer_config: Dictionary containing optimizer class and configuration kwargs
    - self.input_config: Dictionary defining input tensor shapes (e.g., 'x', 'cond', 'time')
    - self.out_dim: Output dimension of the model
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
