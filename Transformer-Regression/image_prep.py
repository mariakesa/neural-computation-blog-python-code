import scipy.io
import matplotlib.pyplot as plt
import numpy as np 
import torch
from transformers import CLIPVisionModel, ViTImageProcessor, ViTModel, AutoProcessor, AutoModel, AutoImageProcessor, ViTMAEModel, ViTForImageClassification


def prepare_imgs():
    imgs = scipy.io.loadmat('/home/maria/Downloads/images_natimg2800_all.mat')['imgs']

    container=[]
    for i in range(0,imgs.shape[2]):
        im=imgs[:,90:180,i]
        container.append(im)

    container=np.array(container)

    imgs=np.transpose(container, (2, 0, 1))

    return imgs


def process_stims(raw_stims, stims, processor, model):
    n_stims=2800
    lst=[]
    for i in range(n_stims):
        print(i)
        inputs = processor(images=stims[i], return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_label = logits.argmax(-1).item()
        print(model.config.id2label[predicted_label])
        lst.append(model.config.id2label[predicted_label])
        plt.imshow(raw_stims[i],cmap='gray')
        plt.title(model.config.id2label[predicted_label],color='orange')
        plt.axis('off')

def make_vit():
    stims=prepare_imgs()
    raw_stims=stims.copy()
    stims = np.repeat(stims[:, np.newaxis, :, :], 3, axis=1)
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch32-384')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch32-384')
    #processor=ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    #model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    embeddings  = process_stims(raw_stims, stims, processor, model)