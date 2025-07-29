import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, PReLU, BatchNormalization, Flatten, UpSampling2D
from keras.layers import LeakyReLU, Dense, Input, add
from keras.applications import VGG19
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

########################################
# Generator Components
########################################

def res_block(ip):
    res = Conv2D(64, (3, 3), padding='same')(ip)
    res = BatchNormalization(momentum=0.5)(res)
    res = PReLU(shared_axes=[1, 2])(res)
    res = Conv2D(64, (3, 3), padding='same')(res)
    res = BatchNormalization(momentum=0.5)(res)
    return add([ip, res])

def upscale_block(ip):
    up = Conv2D(256, (3, 3), padding='same')(ip)
    up = UpSampling2D(size=2)(up)
    up = PReLU(shared_axes=[1, 2])(up)
    return up

def build_generator(input_layer, num_res_blocks=16):
    x = Conv2D(64, (9, 9), padding='same')(input_layer)
    x = PReLU(shared_axes=[1, 2])(x)
    temp = x
    for _ in range(num_res_blocks):
        x = res_block(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.5)(x)
    x = add([x, temp])
    x = upscale_block(x)
    x = upscale_block(x)
    output_layer = Conv2D(3, (9, 9), padding='same')(x)
    return Model(inputs=input_layer, outputs=output_layer)

########################################
# Discriminator Components
########################################

def disc_block(ip, filters, strides=1, bn=True):
    x = Conv2D(filters, (3, 3), strides=strides, padding='same')(ip)
    if bn:
        x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def build_discriminator(input_layer):
    df = 64
    x = disc_block(input_layer, df, bn=False)
    x = disc_block(x, df, strides=2)
    x = disc_block(x, df * 2)
    x = disc_block(x, df * 2, strides=2)
    x = disc_block(x, df * 4)
    x = disc_block(x, df * 4, strides=2)
    x = disc_block(x, df * 8)
    x = disc_block(x, df * 8, strides=2)
    x = Flatten()(x)
    x = Dense(df * 16)(x)
    x = LeakyReLU(alpha=0.2)(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(input_layer, output)

########################################
# VGG for perceptual loss
########################################

def build_vgg(hr_shape):
    vgg = VGG19(weights='imagenet', include_top=False, input_shape=hr_shape)
    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)

########################################
# Combined GAN model
########################################

def build_gan(generator, discriminator, vgg, lr_ip, hr_ip):
    fake_hr = generator(lr_ip)
    discriminator.trainable = False
    valid = discriminator(fake_hr)
    features = vgg(fake_hr)
    return Model(inputs=[lr_ip, hr_ip], outputs=[valid, features])

########################################
# Load Dataset
########################################

def load_images(path, shape):
    image_list = os.listdir(path)[:5000]
    images = []
    for img_name in image_list:
        img = cv2.imread(os.path.join(path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return np.array(images) / 255.

lr_images = load_images("data/lr_images", (32, 32))
hr_images = load_images("data/hr_images", (128, 128))

# Show sample image pair
idx = random.randint(0, len(lr_images) - 1)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.imshow(lr_images[idx]); plt.title("LR")
plt.subplot(1, 2, 2); plt.imshow(hr_images[idx]); plt.title("HR")
plt.show()

# Split data
lr_train, lr_test, hr_train, hr_test = train_test_split(lr_images, hr_images, test_size=0.33, random_state=42)

########################################
# Compile Models
########################################

hr_shape = hr_train.shape[1:]
lr_shape = lr_train.shape[1:]

lr_input = Input(shape=lr_shape)
hr_input = Input(shape=hr_shape)

generator = build_generator(lr_input)
discriminator = build_discriminator(hr_input)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

vgg = build_vgg(hr_shape)
vgg.trainable = False

gan = build_gan(generator, discriminator, vgg, lr_input, hr_input)
gan.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=Adam())

########################################
# Training Loop
########################################

batch_size = 1
epochs = 5
train_lr_batches = [lr_train[i:i+batch_size] for i in range(0, len(lr_train), batch_size)]
train_hr_batches = [hr_train[i:i+batch_size] for i in range(0, len(hr_train), batch_size)]

for epoch in range(epochs):
    g_losses, d_losses = [], []
    for lr_batch, hr_batch in tqdm(zip(train_lr_batches, train_hr_batches), total=len(train_lr_batches)):
        fake_hr = generator.predict_on_batch(lr_batch)
        real_label = np.ones((batch_size, 1))
        fake_label = np.zeros((batch_size, 1))

        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(hr_batch, real_label)
        d_loss_fake = discriminator.train_on_batch(fake_hr, fake_label)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        discriminator.trainable = False
        image_features = vgg.predict(hr_batch)
        g_loss = gan.train_on_batch([lr_batch, hr_batch], [real_label, image_features])

        d_losses.append(d_loss)
        g_losses.append(g_loss[0])

    print(f"Epoch {epoch + 1}: g_loss={np.mean(g_losses):.4f}, d_loss={np.mean(d_losses):.4f}")

    if (epoch + 1) % 10 == 0:
        generator.save(f"gen_e_{epoch + 1}.h5")

########################################
# Evaluation
########################################

generator = load_model('gen_e_10.h5', compile=False)
idx = random.randint(0, len(lr_test))
lr_sample = np.expand_dims(lr_test[idx], axis=0)
hr_sample = np.expand_dims(hr_test[idx], axis=0)

sr_image = generator.predict(lr_sample)

plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1); plt.imshow(lr_sample[0]); plt.title('LR')
plt.subplot(1, 3, 2); plt.imshow(sr_image[0]); plt.title('SR')
plt.subplot(1, 3, 3); plt.imshow(hr_sample[0]); plt.title('HR')
plt.show()

########################################
# Custom Test Image
########################################

sreeni_lr = cv2.imread("data/sreeni_32.jpg")
sreeni_hr = cv2.imread("data/sreeni_256.jpg")
sreeni_lr = cv2.cvtColor(sreeni_lr, cv2.COLOR_BGR2RGB) / 255.
sreeni_hr = cv2.cvtColor(sreeni_hr, cv2.COLOR_BGR2RGB) / 255.
sreeni_lr = np.expand_dims(sreeni_lr, axis=0)
sreeni_hr = np.expand_dims(sreeni_hr, axis=0)

predicted = generator.predict(sreeni_lr)

plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1); plt.imshow(sreeni_lr[0]); plt.title('Input LR')
plt.subplot(1, 3, 2); plt.imshow(predicted[0]); plt.title('Predicted SR')
plt.subplot(1, 3, 3); plt.imshow(sreeni_hr[0]); plt.title('Original HR')
plt.show()
