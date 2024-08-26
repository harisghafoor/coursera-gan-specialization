# import os
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch  # type: ignore
from torch import nn  # type: ignore
from tqdm.auto import tqdm  # type: ignore
from torchvision import transforms  # type: ignore
from torchvision.datasets import MNIST  # type: ignore
from torchvision.utils import make_grid  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import os

torch.manual_seed(0)  # Set for our testing purposes, please do not change!
from scripts.generator import Generator
from scripts.discriminator import Discriminator
from scripts.utils import (
    show_tensor_images,
    get_one_hot_labels,
    combine_vectors,
    get_input_dimensions,
    get_noise,
    weights_init,
    test_input_dims,
)
from scripts.config import Params
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import numpy as np


class Trainer:
    def __init__(self, loss_fn=nn.BCEWithLogitsLoss(), config_params=None):
        self.params = config_params
        self.criterion = loss_fn
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print("Device used for training:", self.device)  # type: ignore
        self.generator_input_dim, self.discriminator_im_chan = get_input_dimensions(
            self.params.z_dim, self.params.mnist_shape, self.params.n_classes
        )
        test_input_dims()
        print("Success!")

    def fit(
        self,
        n_epochs,
        z_dim,
        display_step,
        device,
        n_classes,
        mnist_shape,
        gen,
        disc,
        writer,
        dataloader,
        criterion,
        gen_opt,
        disc_opt,
    ):
        cur_step = 0
        generator_losses = []
        discriminator_losses = []

        # UNIT TEST NOTE: Initializations needed for grading
        noise_and_labels = False
        fake = False

        fake_image_and_labels = False
        real_image_and_labels = False
        disc_fake_pred = False
        disc_real_pred = False

        for epoch in range(n_epochs):
            # Dataloader returns the batches and the labels
            for real, labels in tqdm(dataloader):
                cur_batch_size = len(real)
                # Flatten the batch of real images from the dataset
                real = real.to(device)

                one_hot_labels = (
                    get_one_hot_labels(labels.to(device), n_classes)
                    .squeeze(0)
                    .to(device)
                )
                image_one_hot_labels = one_hot_labels[:, :, None, None]
                image_one_hot_labels = image_one_hot_labels.repeat(
                    1, 1, mnist_shape[1], mnist_shape[2]
                )

                ### Update discriminator ###
                # Zero out the discriminator gradients
                disc_opt.zero_grad()
                # Get noise corresponding to the current batch_size
                fake_noise = get_noise(cur_batch_size, z_dim, device=device)

                # Now you can get the images from the generator
                # Steps: 1) Combine the noise vectors and the one-hot labels for the generator
                #        2) Generate the conditioned fake images

                #### START CODE HERE ####
                noise_and_labels = (
                    combine_vectors(x=fake_noise, y=one_hot_labels)
                    .to(device)
                    .type(torch.float32)
                )
                # print(
                #     noise_and_labels.shape,
                #     type(noise_and_labels),
                #     noise_and_labels.dtype,
                #     noise_and_labels.device,
                #     device,
                # )

                # assert noise_and_labels.device == device
                assert noise_and_labels.dtype == torch.float32
                fake = gen(noise_and_labels)
                #### END CODE HERE ####

                # Make sure that enough images were generated
                assert len(fake) == len(real)

                # Now you can get the predictions from the discriminator
                # Steps: 1) Create the input for the discriminator
                #           a) Combine the fake images with image_one_hot_labels,
                #              remember to detach the generator (.detach()) so you do not backpropagate through it
                #           b) Combine the real images with image_one_hot_labels
                #        2) Get the discriminator's prediction on the fakes as disc_fake_pred
                #        3) Get the discriminator's prediction on the reals as disc_real_pred

                #### START CODE HERE ####

                fake_image_and_labels = combine_vectors(x=fake, y=image_one_hot_labels)
                real_image_and_labels = combine_vectors(x=real, y=image_one_hot_labels)
                disc_fake_pred = disc(fake_image_and_labels)
                disc_real_pred = disc(real_image_and_labels)
                #### END CODE HERE ####

                # Make sure that enough predictions were made
                assert len(disc_real_pred) == len(real)
                # Make sure that the inputs are different
                assert torch.any(fake_image_and_labels != real_image_and_labels)

                disc_fake_loss = criterion(
                    disc_fake_pred, torch.zeros_like(disc_fake_pred)
                )
                disc_real_loss = criterion(
                    disc_real_pred, torch.ones_like(disc_real_pred)
                )
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                disc_loss.backward(retain_graph=True)
                disc_opt.step()

                # Keep track of the average discriminator loss
                discriminator_losses += [disc_loss.item()]

                ### Update generator ###
                # Zero out the generator gradients
                gen_opt.zero_grad()

                fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
                # This will error if you didn't concatenate your labels to your image correctly
                disc_fake_pred = disc(fake_image_and_labels)
                gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
                gen_loss.backward()
                gen_opt.step()

                # Keep track of the generator losses
                generator_losses += [gen_loss.item()]
                #

                # Save metrics
                writer.add_scalar("Loss/mean_discriminator_loss", discriminator_losses[-1], cur_step)  # type: ignore
                writer.add_scalar("Loss/mean_generator_loss", generator_losses[-1], cur_step)  # type: ignore

                ### Visualization code ###
                if cur_step % display_step == 0 and cur_step > 0:
                    fake_fig = show_tensor_images(fake, show=False)
                    real_fig = show_tensor_images(real, show=False)
                    writer.add_figure(f"Images/Fake Images", fake_fig, global_step=cur_step)  # type: ignore
                    writer.add_figure(f"Images/Real Imags", real_fig, global_step=cur_step)  # type: ignore
                    # step_bins = 20
                    # num_examples = (len(generator_losses) // step_bins) * step_bins
                    # plt.plot(
                    #     range(num_examples // step_bins),
                    #     torch.Tensor(generator_losses[:num_examples])
                    #     .view(-1, step_bins)
                    #     .mean(1),
                    #     label="Generator Loss",
                    # )
                    # plt.plot(
                    #     range(num_examples // step_bins),
                    #     torch.Tensor(discriminator_losses[:num_examples])
                    #     .view(-1, step_bins)
                    #     .mean(1),
                    #     label="Discriminator Loss",
                    # )
                    # plt.legend()
                    # plt.show()
                elif cur_step == 0:
                    print(
                        "Congratulations! If you've gotten here, it's working. Please let this train until you're happy with how the generated numbers look, and then go on to the exploration!"
                    )
                cur_step += 1
        writer.close()
        return gen, disc


if __name__ == "__main__":
    conditional_gan = Trainer(loss_fn=nn.BCEWithLogitsLoss(), config_params=Params())
    gen = Generator(input_dim=conditional_gan.generator_input_dim).to(
        conditional_gan.device
    )
    gen_opt = torch.optim.Adam(gen.parameters(), lr=conditional_gan.params.lr)
    disc = Discriminator(im_chan=conditional_gan.discriminator_im_chan).to(
        conditional_gan.device
    )
    disc_opt = torch.optim.Adam(disc.parameters(), lr=conditional_gan.params.lr)
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)
    writer = SummaryWriter(log_dir=conditional_gan.params.log_dir)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataloader = DataLoader(
        MNIST(conditional_gan.params.data_path, download=False, transform=transform),
        batch_size=conditional_gan.params.batch_size,
        shuffle=True,
    )
    gen, disc = conditional_gan.fit(
        n_epochs=conditional_gan.params.n_epochs,
        z_dim=conditional_gan.params.z_dim,
        display_step=conditional_gan.params.display_step,
        device=conditional_gan.device,
        n_classes=conditional_gan.params.n_classes,
        mnist_shape=conditional_gan.params.mnist_shape,
        gen=gen,
        disc=disc,
        writer=writer,
        dataloader=dataloader,
        criterion=conditional_gan.criterion,
        gen_opt=gen_opt,
        disc_opt=disc_opt,
    )
    os.makedirs(conditional_gan.params.save_models_directory, exist_ok=True)
    os.makedirs(conditional_gan.params.log_dir, exist_ok=True)
    torch.save(
        gen.state_dict(),
        conditional_gan.params.save_models_directory + "/mnist_cgan_generator.pth",
    )
    torch.save(
        disc.state_dict(),
        conditional_gan.params.save_models_directory + "/mnist_cgan_discriminator.pth",
    )
    print("Models have been saved")
