import architectures.multifeature_resnet50
import architectures.multifeature_bninception

def select(arch, opt):
    if  'resnet50' in arch:
        return multifeature_resnet50.Network(opt)
    elif  'bninception' in arch:
        return multifeature_bninception.Network(opt)