import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from get_dataset import return_dataset
from models import modded_unet
plt.style.use('dark_background')

MODEL_NAME = "oxford_pets"
classes_dict = {"oxford_pets":3,
                "pascal_voc":21}
IMG_SIZE = 224
N_CLASSES = classes_dict[MODEL_NAME]
EPOCHS = 13
BATCH_SIZE = 12
MODEL_SAVEPATH = "./saved_models/"
TENSORBOARD_LOGDIR = "./tensorboard"

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        #print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

class SegModel:
    def __init__(self, model_path=None):
        if model_path is not None:
            try:
                self.model = self.load_model(model_path)
            except:
                print("\nError loading model. Building model with randomised weights\n")
                self.model = modded_unet.modded_unet(IMG_SIZE, N_CLASSES)
        else:
            self.model = modded_unet.modded_unet(IMG_SIZE, N_CLASSES)
    
    def train_model(self, train_data, savepath, save, val_data=None):
        if val_data is None:
            history = self.model.fit(train_data.batch(BATCH_SIZE),
                                    epochs=EPOCHS,
                                    callbacks=[TensorBoard(log_dir=TENSORBOARD_LOGDIR, write_images=True)],
                                    verbose=1)
        else:
            VAL_SUBSPLITS = 5
            VALIDATION_STEPS = int(val_data.cardinality())//BATCH_SIZE//VAL_SUBSPLITS

            history = self.model.fit(train_data.batch(BATCH_SIZE),
                                    epochs=EPOCHS,
                                    validation_steps=VALIDATION_STEPS,
                                    validation_data=val_data,
                                    callbacks=[TensorBoard(log_dir=TENSORBOARD_LOGDIR, write_images=True)],
                                    verbose=1)
        if save:
            self.save_model(savepath)
        return history

    def predict_datapt_with_model(self, x):
        x = np.reshape(x.numpy(), (1, IMG_SIZE, IMG_SIZE, 3))
        return self.model.predict(x)

    def validate_model(self, val_data, batch_size=128):
        results = self.model.evaluate(val_data.batch(batch_size))
        print("test loss, test acc:", results)
        return results
    
    def sample_prediction(self, data, n=3):
        for img, label in data.take(n):
            pred_label = self.predict_datapt_with_model(img)
            #print(pred_label)

            title = ["image", "label", "prediction"]
            plt.figure(figsize=(25, 25))

            def create_mask(pred_mask):
                pred_mask = tf.argmax(pred_mask, axis=-1)
                pred_mask = pred_mask[..., tf.newaxis]
                return pred_mask[0]

            for i, data in enumerate([img, label, pred_label]):
                plt.subplot(1, 3, i+1)
                #print("\nTitle: %s\n\n"%title[i]+str(np.unique(data, return_counts=True)))
                if i == 0:
                    data = data.numpy()
                    plt.imshow(data)
                elif i == 1:
                    data = data.numpy()
                    plt.imshow(data, cmap=plt.get_cmap('tab20'))
                elif i == 2:
                    mask = create_mask(data)
                    #print("\nMask: \n\n"+str(np.unique(mask, return_counts=True)))
                    plt.imshow(mask, cmap=plt.get_cmap('tab20'))
                plt.axis("off")
                #print(np.array(data).shape)
            plt.tight_layout()
            plt.show()
            
    def save_model(self, model_path):
        self.model.save(model_path)
    
    def load_model(self, path):
        return tf.keras.models.load_model(path)
    
    def plot_train_stats(self, history, results=None):
        plt.figure()
        plt.plot(range(EPOCHS), history.history['loss'], 'r', label='Training loss')
        plt.plot(range(EPOCHS), history.history['accuracy'], 'bo', label='Accuracy')
        if results is not None:
            plt.plot(range(EPOCHS), results.history['loss'], 'g', label='Validation loss')
        plt.title('Training Loss and accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.tight_layout()
        plt.show()

model = SegModel(MODEL_SAVEPATH + MODEL_NAME)
#tf.keras.utils.plot_model(model, show_shapes=True)

train, val, test = return_dataset(MODEL_NAME, IMG_SIZE)
history = model.train_model(train, savepath=MODEL_SAVEPATH + MODEL_NAME, save=True)#, val_data=val)
results = model.validate_model(val)
model.plot_train_stats(history)
model.sample_prediction(test)