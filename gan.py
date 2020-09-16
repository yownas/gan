
import os
import sys
import numpy as np
import cv2
from glob import glob
from matplotlib import pyplot
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

def load_image(image_path):
    img = tf.io.read_file(image_path)
    if IMG_C == 1:
        img = tf.io.decode_jpeg(img, channels=1)
    else:
        img = tf.io.decode_jpeg(img)
    img = tf.image.resize_with_crop_or_pad(img, IMG_H, IMG_W)
    img = tf.cast(img, tf.float32)
    img = (img - 127.5) / 127.5
    return img

def tf_dataset(images_path, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    dataset = dataset.shuffle(buffer_size=10240)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def deconv_block(inputs, num_filters, kernel_size, strides, bn=True):
    x = Conv2DTranspose(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=w_init,
        padding="same",
        strides=strides,
        use_bias=False
        )(inputs)

    if bn:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
    return x


def conv_block(inputs, num_filters, kernel_size, padding="same", strides=2, activation=True):
    x = Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=w_init,
        padding=padding,
        strides=strides,
    )(inputs)

    if activation:
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
    return x

def build_generator(latent_dim):
    f = [2**i for i in range(5)][::-1]
    filters = 32
    output_strides = 16
    h_output = IMG_H // output_strides
    w_output = IMG_W // output_strides

    noise = Input(shape=(latent_dim,), name="generator_noise_input")

    x = Dense(f[0] * filters * h_output * w_output, use_bias=False)(noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((h_output, w_output, 16 * filters))(x)

    for i in range(1, 5):
        x = deconv_block(x,
            num_filters=f[i] * filters,
            kernel_size=5,
            strides=2,
            bn=True
        )

    x = conv_block(x,
        num_filters=IMG_C,
        kernel_size=5,
        strides=1,
        activation=False
    )
    fake_output = Activation("tanh")(x)

    return Model(noise, fake_output, name="generator")

def build_discriminator():
    f = [2**i for i in range(4)]
    image_input = Input(shape=(IMG_H, IMG_W, IMG_C))
    x = image_input
    filters = 64
    output_strides = 16
    h_output = IMG_H // output_strides
    w_output = IMG_W // output_strides

    for i in range(0, 4):
        x = conv_block(x, num_filters=f[i] * filters, kernel_size=5, strides=2)

    x = Flatten()(x)
    x = Dense(1)(x)

    return Model(image_input, x, name="discriminator")

class GAN(Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        for _ in range(2):
            ## Train the discriminator
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            generated_images = self.generator(random_latent_vectors)
            generated_labels = tf.zeros((batch_size, 1))

            with tf.GradientTape() as ftape:
                predictions = self.discriminator(generated_images)
                d1_loss = self.loss_fn(generated_labels, predictions)
            grads = ftape.gradient(d1_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

            ## Train the discriminator
            labels = tf.ones((batch_size, 1))

            with tf.GradientTape() as rtape:
                predictions = self.discriminator(real_images)
                d2_loss = self.loss_fn(labels, predictions)
            grads = rtape.gradient(d2_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        ## Train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as gtape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = gtape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d1_loss": d1_loss, "d2_loss": d2_loss, "g_loss": g_loss}

def save_plot(examples, epoch, n):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        pyplot.subplot(n, n, i+1)
        pyplot.axis("off")

        if IMG_C == 1:
            pyplot.gray()
            img = examples[i].squeeze(axis=-1) # (x, y 1) -> (x, y)
        else:
            img = examples[i]

        pyplot.imshow(img)

    filename = f"samples/epoch-{epoch+1}.png"
    filename = os.path.join(projpath, filename)
    pyplot.savefig(filename)
    pyplot.close()

def usage():
    print("gan.py help | create <project> | train|resume <project> <epochs>")
    sys.exit(1)

if __name__ == "__main__":

    # FIXME
    # Read command line arguments.

    # Help    
    #   help
    # Create stub folders for new project
    #   create <project>
    # Start training new project
    #   train <project> <epochs>
    # Load project and continue (read status.py)
    #   resume <project> <epochs>

    # FIXME
    # Read global config from cwd?

    # Default values
    IMG_H = 64
    IMG_W = 64
    IMG_C = 3
    latent_dim = 128
    batch_size = 32
    num_epochs = 10
    epochs_per_epoch = 1
   
    projdir = "projects"
    n_samples = 1

    try:
        cmd = sys.argv[1]
    except:
        usage()

    if (cmd == "help"):
        usage()
    elif (cmd == "create"):
        try:
            opt1 = sys.argv[2]
        except:
            usage()
	# Create folders and config file
        projpath = os.path.join(projdir, opt1)  
        print("Creating:")
        print("  ",projpath)
        os.makedirs(projpath)
        for subdir in ('data', 'model', 'samples'):
            print("    ",subdir)
            os.makedirs(os.path.join(projpath, subdir))

        # skel -> projects/<project>/config.py
        # touch -> projects/<project>/state.py
        print("  config.py")
        f = open(os.path.join(projpath, "config.py"), "w")
        f.write("# Change things below\n\n")
        f.write("IMG_H = "+str(IMG_H)+"\n")
        f.write("IMG_W = "+str(IMG_W)+"\n")
        f.write("IMG_C = "+str(IMG_C)+" # 1 = Grayscale, 3 = RGB\n\n")
        f.write("batch_size = "+str(batch_size)+"\n")
        f.close()

        print("  state.py")
        f = open(os.path.join(projpath, "state.py"), "w")
        f.write("# No state saved yet.\n")
        f.close()

        print("Done...")
        print("Fill data folder with images, edit config.txt to suit your needs. And then train.")
        sys.exit(0)

    elif (cmd == "train"):
        try:
            opt1 = sys.argv[2]
            opt2 = sys.argv[3]
        except:
            usage()

        projpath = os.path.join(projdir, opt1)  
        exec(open(os.path.join(projpath, "config.py")).read())

        # FIXME
        # Check if there is a state already, refuse to overwrite

        d_model = build_discriminator()
        g_model = build_generator(latent_dim)

        # Make some noise
        noise = np.random.normal(size=(n_samples, latent_dim))
        np.save(os.path.join(projpath, "model", "noise.npy"), noise)
        plot_offset=0

        num_epochs = int(opt2)
        print("Training",opt1,"for",str(num_epochs),"epochs.")

    elif (cmd == "resume"):
        try:
            opt1 = sys.argv[2]
            opt2 = sys.argv[3]
        except:
            usage()

        projpath = os.path.join(projdir, opt1)  
        exec(open(os.path.join(projpath, "config.py")).read())
        exec(open(os.path.join(projpath, "state.py")).read())

        d_model = build_discriminator()
        g_model = build_generator(latent_dim)

        d_model.load_weights(os.path.join(projpath, "model", "d_model.h5"))
        g_model.load_weights(os.path.join(projpath, "model", "g_model.h5"))

        noise = np.load(os.path.join(projpath, "model/noise.npy"))

        num_epochs = int(opt2)
        print("Resume training",opt1,"for",str(num_epochs),"epochs.")

    # ?
    # Show status of project and exit
    #   status <project>
    # Copy project (not data)
    #   copy <project> <new project>

    data = os.path.join(projpath, "data")
    images_path = glob(data+"/*.jpg")
    images_path.extend(glob(data+"/*.png"))

    ############################

    # Read old model & noise? CHANGE PLOT OFFSET
#    if (True):
#    #if (False):
#        plot_offset=200
#        d_model.load_weights("saved_model/d_model.h5")
#        g_model.load_weights("saved_model/g_model.h5")
#        noise = np.load("saved_model/noise.npy")
#    else:
#        # Make some noise
#        noise = np.random.normal(size=(n_samples, latent_dim))
#        np.save("saved_model/noise", noise)
#        plot_offset=0

    #############################

    d_model.summary()
    g_model.summary()

    gan = GAN(d_model, g_model, latent_dim)

    bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    gan.compile(d_optimizer, g_optimizer, bce_loss_fn)

    images_dataset = tf_dataset(images_path, batch_size)

    for epoch in range(num_epochs):
        print("\nEPOCH %i/%i\n" % (epoch, num_epochs))
        gan.fit(images_dataset, epochs=epochs_per_epoch)
        #g_model.save("saved_model/g_model.h5")
        #d_model.save("saved_model/d_model.h5")
        g_model.save(os.path.join(projpath, "model/g_model.h5"))
        d_model.save(os.path.join(projpath, "model/d_model.h5"))

        # Dump status to a status.py file that can be read
        f = open(os.path.join(projpath, "state.py"), "w")
        f.write("# Current epoch: "+str(plot_offset+epoch)+"\n")
        f.write("plot_offset = "+str(plot_offset+epoch)+"\n")
        f.close()

        examples = g_model.predict(noise)
        save_plot(examples, epoch + plot_offset, int(np.sqrt(n_samples)))

