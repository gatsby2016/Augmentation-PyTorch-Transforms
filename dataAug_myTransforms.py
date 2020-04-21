import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

import myTransforms


def main(img, time=0, SAVE=False):
    preprocess = myTransforms.Compose([
        myTransforms.RandomChoice([myTransforms.RandomHorizontalFlip(p=1),
                                   myTransforms.RandomVerticalFlip(p=1),
                                   myTransforms.AutoRandomRotation()]),  # above is for: randomly selecting one for process
        # myTransforms.RandomAffine(degrees=0, translate=[0, 0.2], scale=[0.8, 1.2],
        #                           shear=[-10, 10, -10, 10], fillcolor=(228, 218, 218)),
        myTransforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5)),
        myTransforms.RandomChoice([myTransforms.ColorJitter(saturation=(0, 2), hue=0.3),
                                   myTransforms.HEDJitter(theta=0.05)]),
        # myTransforms.ToTensor(),  #operated on original image, rewrite on previous transform.
        # myTransforms.Normalize([0.6270, 0.5013, 0.7519], [0.1627, 0.1682, 0.0977])
    ])
    print(preprocess)

    preprocess1 = myTransforms.HEDJitter(theta=0.05)
    print(preprocess1)
    preprocess2 = myTransforms.RandomGaussBlur(radius=[0.5, 1.5])
    print(preprocess2)
    preprocess3 = myTransforms.RandomAffineCV2(alpha=0.1)  # alpha \in [0,0.15]
    print(preprocess3)
    preprocess4 = myTransforms.RandomElastic(alpha=2, sigma=0.06, mask=None)
    print(preprocess4)

    composeimg = preprocess(img)
    HEDJitterimg = preprocess1(img)
    blurimg = preprocess2(img)
    affinecvimg = preprocess3(img)
    elasticimg = preprocess4(img)

    if SAVE:
        HEDJitterimg.save('./data/HEDJitter_' + str(time) + '.png')
        blurimg.save('./data/blurimg_' + str(time) + '.png')
        affinecvimg.save('./data/affinecvimg_' + str(time) + '.png')
        elasticimg.save('./data/elasticimg_' + str(time) + '.png')
    else:
        plt.subplot(321)
        plt.imshow(img)
        plt.subplot(322)
        plt.imshow(composeimg)
        plt.subplot(323)
        plt.imshow(HEDJitterimg)
        plt.subplot(324)
        plt.imshow(blurimg)
        plt.subplot(325)
        plt.imshow(affinecvimg)
        plt.subplot(326)
        plt.imshow(elasticimg)
        plt.show()
        plt.close()


if __name__ == '__main__':
    np.random.seed(3)
    random.seed(3)

    img = Image.open('./data/10-05074_353_49_8178.png') # read the image
    print('Raw image shape: ', np.array(img).shape)
    for ind in range(10):
        main(img, time=ind, SAVE=True)
