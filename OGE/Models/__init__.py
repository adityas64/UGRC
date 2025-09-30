# from .ResNet import *
# from .VGG import *
# from .layer import *
from .ResNet_quantize import *
from .VGG_quantize import *
from .layer import *
from models.vgg_light_only_bn import VGG16_light_only_bn


def modelpool(MODELNAME, DATANAME):
    if 'imagenet' in DATANAME.lower():
        num_classes = 1000
    elif '100' in DATANAME.lower():
        num_classes = 100
    else:
        num_classes = 10
    if MODELNAME.lower() == 'vgg16':
        return vgg16(num_classes=num_classes)
    elif MODELNAME.lower() == 'vgg16_light_only_bn':
        return VGG16_light_only_bn(labels=num_classes, dataset=DATANAME)
    elif MODELNAME.lower() == 'vgg19':
        return vgg19(num_classes=num_classes, dropout=0.1)
    elif MODELNAME.lower() == 'vgg16_wobn':
        return vgg16_wobn(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet18':
        return resnet18(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet32':
        return resnet32(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet44':
        return resnet44(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet56':
        return resnet56(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet34':
        return resnet34(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet20':
        return resnet20(num_classes=num_classes)
    else:
        print("still not support this model")
        exit(0)