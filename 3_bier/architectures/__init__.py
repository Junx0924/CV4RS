import architectures.resnet50
import architectures.deep_inception

def select(arch):
    if  'resnet' in arch:
        return resnet50.ResNet50
    elif  'bninception' in arch:
        return deep_inception.GoogleNet