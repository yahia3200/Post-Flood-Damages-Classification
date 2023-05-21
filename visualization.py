import matplotlib.pyplot as plt

def plot_row(images, titles):
    fig, ax = plt.subplots(1, len(images), figsize=(10, 20))
    ax[0].imshow(images[0].astype("uint8"))
    ax[0].set_title(titles[0])
    ax[0].axis("off")
    for i in range(len(images) - 1):
        ax[i+1].imshow(images[i + 1])
        ax[i+1].set_title(titles[i + 1])
        ax[i+1].axis("off")
    plt.tight_layout()
    plt.show()