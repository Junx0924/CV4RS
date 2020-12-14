
import architectures.resnet50
import architectures.googlenet
import architectures.bninception
import architectures.multifeature_resnet50
import architectures.multifeature_bninception

def select(arch, opt):
    if  'multifeature_resnet50' in arch:
        return multifeature_resnet50.Network(opt)
    elif  'multifeature_bninception' in arch:
        return multifeature_bninception.Network(opt)
    elif 'resnet50' in arch:
        return resnet50.Network(opt)
    elif 'googlenet' in arch:
        return googlenet.Network(opt)
    elif 'bninception' in arch:
        return bninception.Network(opt)