from keras.models import model_from_json

def read_model(file_path):
    with open(file_path, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())
    model.compile("adam", "mse")
    weights_file = file_path.replace('json', 'h5')
    model.load_weights(weights_file)
    print("Read existing model from disk")
    return model
