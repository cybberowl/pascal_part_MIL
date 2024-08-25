import matplotlib.pyplot as plt
from .hierarchical_utils import decompose_mask

def plot_examples(X,Y, k = 5, class_content = None, decomposed = False):
    # X and Y are batches from data gen BS x C x H x W
    rows = 2
    if decomposed:
        Y_list = decompose_mask(Y.to('cpu').detach(),class_content)
        rows = rows + len(Y_list) - 1
    for i in range(k):
        plt.subplot(rows, k, i+1)
        plt.axis("off")
        plt.imshow(X[i].to('cpu').permute((1,2,0))) ### C x H x W -> H x W x C
        plt.title('Image')

        if decomposed:
            for j,Y_ in enumerate(Y_list):
                plt.subplot(rows, k, i+k+j*k+1)
                plt.axis("off")
                plt.imshow(Y_[i].to('cpu'), cmap = 'jet')
                plt.title(f'Mask level {j+1}')
        else:
            plt.subplot(rows, k, i+k+1)
            plt.axis("off")
            plt.imshow(Y[i], cmap = 'jet')
            plt.title('Mask')

    plt.show();


def plot_examples_learning(X,true_mask, predicted_masks, k = 3, class_content = None, decomposed = False):
    # X is batch from data gen BS x C x H x W
    # true_mask is batch from data gen BS x 1 x H x W
    # predicted_masks is list of N_LEVELS tensors BS x 1 x H x W
    rows = k
    cols = 3
    if decomposed:
        Y_list = decompose_mask(true_mask.to('cpu').detach(),class_content)
        cols = cols + 2*len(Y_list) - 2
    for i in range(k):
        plt.subplot(rows, cols, i*cols+1)
        plt.axis("off")
        plt.imshow(X[i].to('cpu').permute((1,2,0))) ### C x H x W -> H x W x C
        plt.title(f'Image #{i}')

        if decomposed:
            for j,Y_ in enumerate(Y_list):
                plt.subplot(rows, cols, i*cols+2+j)
                plt.axis("off")
                plt.imshow(Y_[i].to('cpu'), cmap = 'jet')
                plt.title(f'True L{j+1} #{i}')

                plt.subplot(rows, cols, i*cols+2+j + len(Y_list))
                plt.axis("off")
                plt.imshow(predicted_masks[j][i].to('cpu'), cmap = 'jet')
                plt.title(f'Predicted L{j+1} #{i}')
        else:
            plt.subplot(rows, cols, i*cols+2)
            plt.axis("off")
            plt.imshow(true_mask[i].to('cpu'), cmap = 'jet')
            plt.title(f'True #{i}')

            plt.subplot(rows, cols, i*cols+3)
            plt.axis("off")
            plt.imshow(predicted_masks[-1][i].to('cpu'), cmap = 'jet')
            plt.title(f'Predicted #{i}')

    plt.tight_layout()
    plt.show();