import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import io

def prepare_imgs():
    imgs = scipy.io.loadmat('/home/maria/Downloads/images_natimg2800_all.mat')['imgs']
    container = []
    for i in range(0, imgs.shape[2]):
        im = imgs[:, 90:180, i]
        container.append(im)
    container = np.array(container)
    imgs = np.transpose(container, (2, 0, 1))
    return container

def process_stims(raw_stims, stims, processor, model, save_path='output.gif'):
    n_stims = 10  # Adjusted for testing, you can set it back to 2800
    frames = []
    for i in range(n_stims):
        print(i)
        inputs = processor(images=stims[i], return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        print(model.config.id2label[predicted_label])
        # Plotting the image
        plt.imshow(raw_stims[i], cmap='gray')
        plt.title(r'\textcolor{blue}{Vision Transformer vit-base-patch32-384 model prediction:}' + '\n' +
                  r'\textcolor{green}{' + model.config.id2label[predicted_label] + '}', fontsize=15, loc='center', usetex=True)
        plt.axis('off')
        # Saving the image to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        # Append the image to the list of frames
        frames.append(Image.open(buf))
        buf.close()
        plt.close()

    # Create a GIF from the frames
    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=500, loop=0)

def make_vit():
    stims = prepare_imgs()
    raw_stims = stims.copy()
    stims = np.repeat(stims[:, np.newaxis, :, :], 3, axis=1)
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch32-384')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch32-384')
    process_stims(raw_stims, stims, processor, model)

make_vit()
