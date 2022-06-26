import torch
import torch
from architecture import vae
from trainNtest import training, testing
from utility import cost_graph, view_images, plot_latent
import loader

batch_size = 64
num_epochs = 5
training_data = loader.trainLoader(batch_size)
testing_data = loader.testLoader(batch_size)


def main(training_data, testing_data, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = vae().to(device)
    opt = model.optimizer()
    train_loss = []
    test_loss = []
    train_output = []
    test_output = []
    latent_space = []
    FILE = "VAE_model.pth"
    ch = input("Press l to load model, t to train model: ").lower()  # asks user if they want to train the model or load the already saved model
    if ch == 'l':
        model.load_state_dict(torch.load(FILE))  # loads the model
        # test the loaded model on the Test data
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}")
            test_epoch_loss = testing(model, testing_data, epoch, test_output, device)
            print(f"Test Loss: {test_epoch_loss}")
            test_loss.append(test_epoch_loss)
    elif ch == 't':
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}")
            train_epoch_loss, latent_space = training(model, training_data, opt, epoch, train_output, latent_space, device, num_epochs)
            test_epoch_loss = testing(model, testing_data, epoch, test_output, device)
            print(f"Train Loss: {train_epoch_loss} \t Test Loss: {test_epoch_loss}")
            train_loss.append(train_epoch_loss)
            test_loss.append(test_epoch_loss)

        plot_latent(latent_space)
        cost_graph(train_loss, "VAE Train Loss")
        cost_graph(test_loss, "VAE Test Loss")
        view_images(train_output, num_epochs)
        torch.save(model.state_dict(), FILE)  # saves the trained model at the specified path


    # files.download('VAE_model.pth') colab

    view_images(test_output, num_epochs)


main(training_data, testing_data, num_epochs)
