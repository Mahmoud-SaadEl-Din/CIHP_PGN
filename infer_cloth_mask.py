import PIL.Image as Image
import os
import numpy as np
from carvekit.api.interface import Interface
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub
from carvekit.trimap.generator import TrimapGenerator


# Check doc strings for more information
seg_net = TracerUniversalB7(device='cpu',
            batch_size=1)

fba = FBAMatting(device='cpu',
                input_tensor_size=100,
                batch_size=1)

trimap = TrimapGenerator()

preprocessing = PreprocessingStub()

postprocessing = MattingMethod(matting_module=fba,
                            trimap_generator=trimap,
                            device='cpu')

interface = Interface(pre_pipe=preprocessing,
                    post_pipe=postprocessing,
                    seg_pipe=seg_net)


def infere_cloth_mask(root, out):

    imgs = []
    for name in os.listdir(root):
        imgs.append(root + '/' + name)


    def get_bg_indx(image):
        idx = (image[...,0]==130)&(image[...,1]==130)&(image[...,2]==130) # background 0 or 130, just try it
        if idx[0][0] == False:
            idx = (image[...,0]==0)&(image[...,1]==0)&(image[...,2]==0) # background 0 or 130, just try it
        return idx
    
    images = interface(imgs)
    for i, im in enumerate(images):
        img = np.array(im)
        img = img[...,:3] # no transparency
        idx = get_bg_indx(img)
        img = np.ones(idx.shape)*255
        img[idx] = 0
        im = Image.fromarray(np.uint8(img), 'L')
        im_name = f'{out}/{imgs[i].split("/")[-1].split(".")[0]}.png'
        print(im_name)
        im.save(im_name)

    return len(images)