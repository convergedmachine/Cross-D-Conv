from torchvision.models import resnet18
from replace_conv_layers import replace_conv_layers

def create_model(args):
    print("Creating model")
    my_model = resnet18(num_classes=args.num_classes)
    if args.rot is not None:
        replace_conv_layers(my_model) #, args.rot)
    return my_model