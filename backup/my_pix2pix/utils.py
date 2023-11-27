import torch, numpy
import config
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    # x = x.transpose(1,2,0)
    # y = y.transpose(1,2,0)
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        input_image = x
        transform = transforms.ToTensor()
        x = transform(x).unsqueeze(0)
        y_fake = gen(x)

        # Convert the output tensor to a NumPy array
        output_array = y_fake.squeeze().cpu().numpy()

        # Transpose the output array back to (length, breadth, channels) format
        output_array = numpy.transpose(output_array, (1, 2, 0))

        # Convert the output array to a PIL image
        output_image = Image.fromarray((output_array).astype(numpy.uint8))
        input_image = Image.fromarray((input_image).astype(numpy.uint8))

        # Save the output image as a .png file
        output_image.save(folder + f"/y_gen_{epoch}.png")
        input_image.save(folder + f"/input_{epoch}.png")

        # save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        # save_image(x, folder + f"/input_{epoch}.png")
        # if epoch == 1:
        #     save_image(y, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

