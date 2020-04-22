# Augmentation-PyTorch-Transforms
**Image data augmentation on-the-fly by adding new class on transforms in PyTorch and torchvision.**

> Normally, we `from torchvision import transforms` for transformation, but some specific transformations (especially for **histology** image augmentation) are missing.  

> Thus, we add 4 new transforms class on the basic of `torchvision.transforms` pyfile, which we named as [myTransforms.py](https://github.com/gatsby2016/Augmentation-PyTorch-Transforms/blob/master/myTransforms.py).  
  
> You can call and use it in the same form as `torchvision.transforms`. Or, you can refer to [dataAug_myTransforms.py](https://github.com/gatsby2016/Augmentation-PyTorch-Transforms/blob/master/dataAug_myTransforms.py).  

> Also, you can check the actual effect of [myTransforms](https://github.com/gatsby2016/Augmentation-PyTorch-Transforms/blob/master/myTransforms.py) for data augmentation :)  


## New transforms classes included in `myTransforms`
### **HEDJitter**
Randomly perturbe the HED color space value on an RGB **pathological** image[1].
1. Disentangle the hematoxylin and eosin color channels by color deconvolution[2] method using a fixed matrix.
2. Perturbe the hematoxylin, eosin and DAB stains independently.
3. Transform the resulting stains into regular RGB color space.

**Args**    
- theta (float): How much to jitter HED color space,  
- then, alpha is chosen from a uniform distribution [1-theta, 1+theta]  
- betti is chosen from a uniform distribution [-theta, theta]  
- the jitter formula is $s' = \alpha * s + \betti$  
  
**Example**  
```python
import myTransforms
imagename = '../data/10-05074_353_49_8178.png'
img = Image.open(imagename) # read the image
	
preprocess = myTransforms.HEDJitter(theta=0.05)
print(preprocess)
	
HEPerimg = preprocess(img)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(HEPerimg)
plt.show()
```
![HEDjitter](https://github.com/gatsby2016/Augmentation-PyTorch-Transforms/blob/master/data/HEDJitter.gif)

	
**References**    
[1]. [Tellez, D., Balkenhol, M., Otte-Höller, I., van de Loo, R., Vogels, R., Bult, P., ... & Litjens, G. (2018). Whole-slide mitosis detection in H&E breast histology using PHH3 as a reference to train distilled stain-invariant convolutional networks. IEEE transactions on medical imaging, 37(9), 2126-2136.](https://ieeexplore.ieee.org/abstract/document/8327641)  
[2]. [Ruifrok, A. C., & Johnston, D. A. (2001). Quantification of histochemical staining by color deconvolution. Analytical and quantitative cytology and histology, 23(4), 291-299.](https://s3.amazonaws.com/academia.edu.documents/40705455/AnalQuantCytHist-AR.pdf?response-content-disposition=inline%3B%20filename%3DQuantification_of_histochemical_staining.pdf&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIATUSBJ6BAPUHRL6A6%2F20200421%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200421T015610Z&X-Amz-Expires=3600&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEPn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQCrEZlRwVoTK5%2Fx14jlpODaSkPreJ5NY37x1uTyYAPlXQIgBZfB%2FpM4Inbb9nrIV%2BO7UCpH7m2B4xRky7dkoHmLy9sqtAMIIhAAGgwyNTAzMTg4MTEyMDAiDOVzPPqXY51w8HhKQSqRAzYnVcq6y8NUQRflE9cOTUiZ17Uh16MWC03iOPCGdSlBQBxyAwzQNqYrPFsX0WijlfQ1NsouAKa09n1syHbUMHQmczA52NthM1LVjQS%2FaB4e2QR%2B9JBNQz6Y1M2rvyy1IOcpouc%2Bb%2B4Pbl8zPxnzyxmzDf5e0VwB1l0F%2BqhQG0HCZxY8K7GssUqLaOempoDOFnpfe21HMyE3hrOqGMdA4Rp7ThBHoODkNpZc9je4v%2ByX97%2BCrOMZGX4Qrc4ZNVGk0ku9N5ly75h2qB3gsnnFmhATkKqRxNGQpqtGAPPPxy%2B4C77%2Fgeds%2Bu9v4C1dAaEwqu%2Fca%2Fc871PlZV42vLus%2FX%2FGwYcEH5tOUNdGHTDKDqPhPtS8fzKYX%2F2Q9JvogD5lTeGXtPXWJvOnGH%2F%2BNOQlRb3i3w0GWx%2BiB7AjMJVEvY9jjm3iDp2BZsxLWIc9dlX7CDKAz5qcoEd3XMsoiXhCDlwHOvx5VqDYICNrqBVRvMo5cPGui2KmoISmmdeNuyjqhBJ9%2FVq4cm1v72NT%2BIXhStHLMOaD%2BfQFOusB7%2B%2FnDDqzMUvjiFC7X1pgtrmXaTfkhMv21SGJyTvwvwPtGNh2qjBApe2dkfyFYn%2F%2BLDcVCV0574Kv3RBVTQcUd20ea1H2ZfgaCbuLFl4nhbMxbjnQgmF5anccbyCJfhHxsCWCgHZRfZR%2BwEDqMGREdHkx4R5gi5g6rsTs0iIuGM2aKpNWK%2BtBXjteH7JK1rWI6GQIxvclg2HmM3ET9gHqiirbMhVemadRloQjRjgAhYafnIu5n%2FqnbPfcDCVBlo7y2I6IofZtjrFZDfU59RS3xeqYCCunfmMvu3XOp7RzJutxEqgC4iMwCXOlLA%3D%3D&X-Amz-SignedHeaders=host&X-Amz-Signature=da6b0b590af1cc2982b418350faae562741bfea920a606337c1e3487adc24d61)

	
### **RandomElastic**
Random Elastic transformation by *CV2* method on image by alpha, sigma parameter.

**Args**      
- alpha (float): alpha value for Elastic transformation, factor on dx, dy  
-- if alpha is 0, output is the same as origin, whatever the sigma;  
-- if alpha is 1, output only depends on sigma parameter;  
-- if alpha < 1 or > 1, it zoom in or out the sigma-relevant dx, dy.  
- sigma (float): sigma value for elastic transformation, should be $\in (0.05,0.1)$  
- mask (PIL Image) For processing on GroundTruth of segmentation task, if not assign, set None.  

**Example**  
```python
import myTransforms
imagename = '../data/10-05074_353_49_8178.png'
img = Image.open(imagename) # read the image
	
preprocess = myTransforms.RandomElastic(alpha=2, sigma=0.06, mask=None)
print(preprocess)
	
elasticimg = preprocess(img)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(elasticimg)
plt.show()
```		
![Elastic](https://github.com/gatsby2016/Augmentation-PyTorch-Transforms/blob/master/data/elasticimg.gif)
  

**References**    
[affine and elastic transform](https://blog.csdn.net/maliang_1993/article/details/82020596)  
[cv2.warpAffine](https://blog.csdn.net/qq_27261889/article/details/80720359)  
[scipy.ndimage.map_coordinates](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html#scipy.ndimage.map_coordinates)  

	
### RandomAffineCV2
Random Affine transformation by CV2 method on image by alpha parameter.  
It is different from `torchvision.transforms.RandomAffine`, which is implemented by `PIL.Image` method. We can set BORDER_REFLECT for the area outside the transform in the output image while original `RandomAffine` can only fill by a specified value.  

**Args**     
- alpha (float): alpha value for affine transformation  
- mask (PIL Image) For processing on GroundTruth of segmentation task, if not assign, set None.  
	
**Example**  
```python
import myTransforms
imagename = '../data/10-05074_353_49_8178.png'
img = Image.open(imagename) # read the image
	
preprocess = myTransforms.RandomAffineCV2(alpha=0.1)#alpha \in [0,0.15]
print(preprocess)
	
affinecvimg = preprocess(img)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(affinecvimg)
plt.show()
```
![RandomAffineCV2](https://github.com/gatsby2016/Augmentation-PyTorch-Transforms/blob/master/data/affinecvimg.gif)


### RandomGaussBlur
Random Gauss Blurring on image by radius parameter.
**Args**  
- radius (list, tuple): radius range for selecting from; you'd better set it < 2 especially for histopathological image task.  

**Example**  
```python
import myTransforms
imagename = '../data/10-05074_353_49_8178.png'
img = Image.open(imagename) # read the image
	
preprocess = myTransforms.RandomGaussBlur(radius=[0.5, 1.5])
print(preprocess)
	
blurimg = preprocess(img)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(blurimg)
plt.show()
```
![GaussianBlur](https://github.com/gatsby2016/Augmentation-PyTorch-Transforms/blob/master/data/blurimg.gif)


### AutoRandomRotation
change `torchvision.transforms.RandomRotation` for auto-random select angle from [0, 90, 180, 270] for rotating the image.

**Example**
```python
import myTransforms
imagename = '../data/10-05074_353_49_8178.png'
img = Image.open(imagename) # read the image
	
preprocess = myTransforms.AutoRandomRotation()
print(preprocess)
	
rotateimg = preprocess(img)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(rotateimg)
plt.show()
```
