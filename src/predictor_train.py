import logging 
import os 
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from models.model_trainer import ModelTrainer

def main():
    MT = ModelTrainer()
    MT.run()

if __name__ == "__main__":
    main()