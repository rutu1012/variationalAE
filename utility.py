from matplotlib import pyplot as plt


def cost_graph(loss_list, title):
    plt.plot(loss_list)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title(title)
    plt.show()


def plot_latent(latent_space):
    for z, y in latent_space:
        z = z.cpu().detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
    plt.colorbar()


def view_images(output, num_epochs):

    for k in range(0, num_epochs, num_epochs//3):

        ori_imgs = output[k][1]
        recon = output[k][2]
        num = 10

        plt.figure(figsize=(18, 5))
        plt.suptitle("Epoch: %i" % (k + 1))

        for i in range(num):
            # plt.subplot(2, n, i+1)
            ax = plt.subplot(2, num, i + 1 + num)
            plt.imshow(recon[i].cpu().detach().numpy().reshape(28, 28), cmap="gray")
            plt.xticks([])
            plt.yticks([])  # removing axes


        plt.show()
