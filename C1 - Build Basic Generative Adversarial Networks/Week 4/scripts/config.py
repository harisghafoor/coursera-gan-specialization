class Params:
    def __init__(self):
        self.image_size = 28
        self.mnist_shape = (1, self.image_size, self.image_size)
        self.n_classes = 10
        self.n_epochs = 100
        self.z_dim = 64
        self.display_step = 500
        self.batch_size = 128
        self.lr = 0.0002
        self.dataset_name = "MNIST" # or Celeba

        if self.dataset_name == "MNIST":
            self.data_path = "/Users/eloise-em/Documents/GitHub/coursera-gan-specialization/C1 - Build Basic Generative Adversarial Networks/Week 1/data/"
        elif self.dataset_name == "Celeba":
            self.data_path = "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba"
        self.experiment_name = f"Conditional-GAN-CLI-Testing-{self.dataset_name}"
        self.log_dir=f"/Users/eloise-em/Documents/GitHub/coursera-gan-specialization/runs" + "/" + self.experiment_name
        self.save_models_directory = "/Users/eloise-em/Documents/GitHub/coursera-gan-specialization/models"  + "/" + self.experiment_name