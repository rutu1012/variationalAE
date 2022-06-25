from matplotlib import pyplot as plt
def cost_graph(loss_list,title):
    plt.plot(loss_list)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title(title)
    plt.show()

def plot_latent(latent_space):
  for z,y in latent_space:
    z = z.cpu().detach().numpy() 
    plt.scatter(z[:, 0],z[:, 1],c=y,cmap='tab10')
  plt.colorbar()

def view_images(output,num_epochs):
    # print(np.array(output).shape)
    for k in range(0,num_epochs):
        # plt.figure(figsize=(18, 5))
        # plt.gray()
        ori_imgs = output[k][1]
        recon = output[k][2]
        n= 10
        print("Epoch ",k)
        # print(recon.shape)
        plt.figure(figsize=(20, 5))
        for i in range(n):
            plt.subplot(2, n, i+1)
            # # plt.title("Original")
            # # plt.imshow(ori_imgs[i].cpu().detach().numpy().reshape(28,28),cmap = "gray")
            # plt.subplot(2, n, i+1+n)
            # plt.title("VAE")
            plt.imshow(recon[i].cpu().detach().numpy().reshape(28,28),cmap = "gray")
        # plt.suptitle("Reconstructed")
        # plt.imshow(torchvision.utils.make_grid(recon[i].reshape(28,28)))
        plt.show()
