import architectures.multifeature_resnet50
import architectures.multifeature_bninception

def select(arch, opt):
    if  'multifeature_resnet50' in arch:
        return multifeature_resnet50.Network(opt)
    if  'multifeature_bninception' in arch:
        return multifeature_bninception.Network(opt)