
from models.grad_cam import GradCAM
from utils.image import show_cam_on_image
from torchvision.models import resnet50

def audioMask(net_classifyOnAudio, audioFeature, label):

    model = net_classifyOnAudio
    target_layer = model.layer5[-1]
    input_tensor = audioFeature# Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=False)

    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    target_category = label  # list [X]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # [X,16,16]





    # In this example grayscale_cam has only one image in the batch:
    # grayscale_cam = grayscale_cam[0, :]
    # visualization = show_cam_on_image(rgb_img, grayscale_cam)



