from train import main
import config

def custom_train():
    while config.RIGHT <= 90:
        main()
        config.LEFT += 15
        config.RIGHT += 15

if __name__ == "__main__":
    custom_train()