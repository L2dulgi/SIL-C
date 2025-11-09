class BaseInterface:
    def __init__(self):
        self.entry_skill_map = None
        self.prior = None # aux model for skill prior 
        pass

    def init_entry(self, dataloader):
        """Initialization step: Create or reset necessary fields if needed."""
        raise NotImplementedError

    def map_entry(self, dataloader):
        """Main mapping logic: define subgoals, cluster them, and populate fields."""
        raise NotImplementedError

    def create_interface(self, dataloader):
        """Entry point that orchestrates the interface processing."""
        raise NotImplementedError

    def update_interface(self, dataloader):
        """Update the interface with new data."""
        raise NotImplementedError

    # dataloader update functions.   
    def update_dataloader(self, dataloader) :
        """Update the dataloader with new data."""
        raise NotImplementedError
    
    def rollback_dataloader(self, dataloader) :
        """Rollback the dataloader to the previous state."""
        raise NotImplementedError

    def forward(self, entry):
        """Forward pass of the interface."""
        raise NotImplementedError

    def reset(self):
        """Reset the interface to its initial Zstate."""
        self.entry_skill_map = None
        self.prior = None