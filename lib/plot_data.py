import matplotlib.pyplot as plt


def plot(x_train):
    images = [x_train[i] for i in range(15)]
    fig, axes = plt.subplots(3, 5, figsize = (10, 10))
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()