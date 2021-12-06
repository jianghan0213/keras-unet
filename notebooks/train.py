import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from PIL import Image

masks = glob.glob(r"C:\Users\92035\Downloads\train_512_5_no_overlap_dilation_125_rect_color_aug_10x\label\*.png")
orgs = glob.glob(r"C:\Users\92035\Downloads\train_512_5_no_overlap_dilation_125_rect_color_aug_10x\image\*.png")

imgs_list = []
masks_list = []
for image, mask in zip(orgs, masks):
    imgs_list.append(np.array(Image.open(image).resize((512, 512))))

    im = Image.open(mask).resize((512, 512))
    # width, height = im.size   # Get dimensions

    # left = (width - 388)/2
    # top = (height - 388)/2
    # right = (width + 388)/2
    # bottom = (height + 388)/2

    # im_cropped = im.crop((left, top, right, bottom))
    masks_list.append(np.array(im))

imgs_np = np.asarray(imgs_list)
masks_np = np.asarray(masks_list)

from keras_unet.utils import plot_imgs

plot_imgs(org_imgs=imgs_np, mask_imgs=masks_np, nm_img_to_plot=10, figsize=6)

# masks_np = np.stack((masks_np, masks_np, masks_np), axis=3)

x = np.asarray(imgs_np, dtype=np.float32)/255
y = np.asarray(masks_np, dtype=np.float32)

y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

from keras_unet.utils import get_augmented

train_gen = get_augmented(
    x_train, y_train, batch_size=2,
    data_gen_args = dict(
        rotation_range=15.,
        # width_shift_range=0.05,
        # height_shift_range=0.05,
        # shear_range=50,
        # zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant'
    ))


from keras_unet.models import custom_unet, attention_ResUNet

input_shape = x_train[0].shape

print(input_shape)

# model = custom_unet(
#     input_shape,
#     use_batch_norm=False,
#     num_classes=2,
#     filters=64,
#     dropout=0.2,
#     output_activation='sigmoid'
# )

model = attention_ResUNet(
    input_shape,
    use_batch_norm=False,
    num_classes=1,
    filters=64,
    dropout=0.2,
    output_activation='sigmoid'
)

import os
os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\Graphviz2.38\\bin\\"

import sys
print(sys.path)

sys.path.append("C:\\Program Files (x86)\\Graphviz2.38\\bin\\")

print(sys.path)

from keras.callbacks import ModelCheckpoint


model_filename = 'segm_model_v0.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename,
    verbose=1,
    monitor='val_loss',
    save_best_only=True,
)
model.summary()


from tensorflow.keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance

model.compile(
    #optimizer=Adam(),
    optimizer=SGD(lr=0.01, momentum=0.99),
    loss='binary_crossentropy',
    #loss=jaccard_distance,
    metrics=[iou, iou_thresholded]
)

history = model.fit_generator(
    train_gen,
    steps_per_epoch=100,
    epochs=10,

    validation_data=(x_val, y_val),
    callbacks=[callback_checkpoint]
)

from keras_unet.utils import plot_segm_history

plot_segm_history(history)

model.load_weights(model_filename)
y_pred = model.predict(x_val)



from keras_unet.utils import plot_imgs

plot_imgs(org_imgs=x_val, mask_imgs=y_val, pred_imgs=y_pred, nm_img_to_plot=9)