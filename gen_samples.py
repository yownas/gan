
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from matplotlib import pyplot

def save_plot(examples, n, idx):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        pyplot.subplot(n, n, i+1)
        pyplot.axis("off")
        pyplot.imshow(examples[i])
    filename = "anim/fake-%d.png" % (idx)
    pyplot.savefig(filename)
    pyplot.close()

if __name__ == "__main__":
    model = load_model("model/g_model.h5")

    #n_samples = 4     ## n should always be a square of an integer.
    latent_dim = 128
    #latent_points = np.random.normal(size=(n_samples, latent_dim))

    faces = 50
    steps = 50

    idx = 0

    start_face = np.random.normal(size=(1, latent_dim))
    for f in range(faces):
    
        # Faces
        stop_face = np.random.normal(size=(1, latent_dim))
        diff_face = np.empty(shape=(1, latent_dim))
        for i in range(len(start_face)):
            diff_face[i] = -(start_face[i]-stop_face[i])/steps

        for s in range(steps):
            idx += 1
            print(idx)
            examples = model.predict(start_face)
            save_plot(examples, 1, idx)
            #save_plot(examples, int(np.sqrt(n_samples)), idx)

            for i in range(len(start_face)):
                start_face[i] += diff_face[i]

        start_face = stop_face
