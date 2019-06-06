from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import datetime
import random
import numpy as np
from SrganModel import SRGAN
from DataLoader import DataLoader
import matplotlib.pyplot as plt

####################################################
################ Configure Model ###################
####################################################

def cMSE(hr, sr):

    # apply the quality mask
    obs = tf.equal(hr, 0.05)
    clr = tf.math.logical_not(obs)
    _hr = tf.boolean_mask(hr, clr )
    _sr = tf.boolean_mask(sr, clr )

    # calculate the bias in brightness b
    pixel_diff = _hr - _sr
    b = K.mean(pixel_diff)

    # calculate the corrected clear mean-square error
    pixel_diff -= b
    cMse = K.mean(pixel_diff * pixel_diff)

    return cMse

# instantiate modules
Srgan = SRGAN()
data_loader = DataLoader()

# define optimizers
discr_optimizer = Adam(lr=1e-3, clipvalue=1.0, decay=1e-8)
gan_optimizer = Adam(lr=1e-4, clipvalue=1.0, decay=1e-8)

# initiate generator
generator = Srgan.generator()

# initiate and complie discriminator
discriminator = Srgan.discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=discr_optimizer)

# initiate and compile the GAN model
srgan = Srgan.srgan(generator, discriminator)
srgan.compile(loss=[cMSE, 'binary_crossentropy'],
              loss_weights=[6*1e-3, 1e-3],
              optimizer=gan_optimizer)

###################################################
################ Vizualization  ###################
###################################################

def plot_generated_images(epoch, generator, iteration, num_pass, examples=1 , dim=(1, 3),
                         save=True):

    imgs_hr, imgs_lr = data_loader.load_data(examples)
    gen_img = generator.predict(imgs_lr)

    fig = plt.figure(figsize=(15, 5))

    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(np.squeeze(imgs_lr[0]), interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(np.squeeze(gen_img[0]), interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(np.squeeze(imgs_hr[0]), interpolation='nearest')
    plt.axis('off')

    plt.tight_layout()
    if save:
        fig.savefig("output/predict{}_{}_{}.png".format(num_pass+1, epoch+1, iteration+1))
        plt.close(fig)


###################################################
################ Training SRGAN  ##################
###################################################

def train(epochs=20000, batch_size=10, sample_interval=200, num_pass=0):

    start_time = datetime.datetime.now()

    for epoch in range(epochs):

        d_losses = []
        g_losses = []
        data_len = 1160

        print ('-'*15, 'Epoch %d' % (epoch+1), '-'*15)
        for iteration in range(data_len//batch_size):

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample lr nad hr images
            imgs_hr, imgs_lr = data_loader.load_data(batch_size)

            # Create fake hr image using generator (and their labels)
            fake_hr = generator.predict(imgs_lr)
            valid = np.ones(batch_size) - np.random.random(batch_size)*0.2
            fake = np.random.random(batch_size)*0.2

            discriminator.trainable = True

            # Train the discriminator (original images = real / generated = Fake)
            d_loss_real = discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = discriminator.train_on_batch(fake_hr, fake)
            d_losses.append(0.5 * np.add(d_loss_real, d_loss_fake))

            # ------------------
            #  Train Generator
            # ------------------

            # Sample lr nad hr images
            imgs_hr, imgs_lr = data_loader.load_data(batch_size)

            # The generator want the discriminator to label the generated images as real
            valid = np.ones(batch_size) - np.random.random(batch_size)*0.2

            discriminator.trainable = False

            # Train the generator
            g_loss = srgan.train_on_batch(imgs_lr, [imgs_hr, valid])
            g_losses.append(g_loss[0])

            elapsed_time = datetime.datetime.now() - start_time

            # Print and plot the progress
            print("iteration {}/{} in epoch {}/{}".format(iteration+1,
                                                    data_len//batch_size,
                                                    epoch+1,
                                                    epochs))
            print ("         time:           %s" % (elapsed_time))

            if iteration == 0 or iteration % sample_interval == 0:
                plot_generated_images(epoch, generator, iteration, num_pass)

        # Save model after each epoch
        generator.save('output/gen_model{}_{}.h5'.format(num_pass+1, epoch+1))
        discriminator.save('output/dis_model{}_{}.h5'.format(num_pass+1,epoch+1))
        srgan.save('output/gan_model{}_{}.h5'.format(num_pass+1,epoch+1))

        print(
        f'discriminator loss = {np.mean(d_losses):.5f} '
        f'generator loss = {np.mean(g_losses):.5f} ')


if __name__ == '__main__':

    # ------------------------------------------------------------------------
    #  These parameters are set to train for 1000 epochs, with only 8GB of GPU
    #  VRAM available.
    #  It is recommended to train for more than that, and to increase the batch
    #  size if you have appropriate GPU ressources.
    # ------------------------------------------------------------------------

    pass_num = 0
    while pass_num<200:
        train(epochs=5, batch_size=2, sample_interval=100, num_pass = pass_num)
        pass_num+=1
        print("#"*80)
        print("#"*35," {} Overhaul Epochs Completed! ".format(5*pass_num),"#"*35)
        print("#"*80)
        time.sleep(100 + random.randint(1,50))
