import pickle
import numpy as np
from SILGym.utils.logger import get_logger

logger = get_logger(__name__)

path = " "
with open(path, 'rb') as f:
    data = pickle.load(f)
    
    for k, i in data.items():
        logger.info(k)
        logger.info(i.shape)
        logger.info("="*50)
        break
