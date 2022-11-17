from keras.models import load_model
import sys

if __name__=="__main__":
    model_name = sys.argv[1]
    print("loading {}".format(model_name))
    model = load_model(sys.argv[1])
    print(model.summary())