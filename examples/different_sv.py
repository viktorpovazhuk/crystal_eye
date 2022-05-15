if DEBUG_K:
    divider = 20
    plt_num = k // divider

    fig, axs = plt.subplots(int((plt_num + 1) / 2), 2, figsize=(30, 30))
    plt.subplots_adjust(wspace=0.3, hspace=0.2)

    for i in range(plt_num):
        j = (i + 1) * divider

        deb_mat = U[:, :j] @ Sigma[:j, :j] @ VT[:j, :]

        axs[i // 2][i % 2].imshow(deb_mat, cmap='gray')
        axs[i // 2][i % 2].set_title(f'j: {j}')

    fig.savefig('data/multiple_plots.png')

    return