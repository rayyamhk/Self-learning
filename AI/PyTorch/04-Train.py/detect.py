import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])

def detect(model, img_path, labels_mapping):
    model.eval()
    softmax = torch.nn.Softmax(dim=1)

    raw_img = Image.open(img_path)
    img = transform(raw_img)
    img = img.view(1, 3, 224, 224)

    pred = model(img)
    pred = softmax(pred)
    arg = torch.argmax(pred, dim=1)
    label = labels_mapping[arg]

    plt.title('Predict: %s' % (label))
    plt.imshow(raw_img)
    plt.show()

    return label
