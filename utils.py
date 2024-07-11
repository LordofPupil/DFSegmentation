import glob
import os
import pickle
import re
import matplotlib.pyplot as plt


def natural_sort_key(s):
    sub_strings = re.split(r'(\d+)', s)
    sub_strings = [int(c) if c.isdigit() else c for c in sub_strings]
    return sub_strings


def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f)


def write_to_file(file_path, content):
    try:
        with open(file_path, 'a') as file:
            file.write(content + '\n')
            print("Content write in file successfully！")
    except IOError:
        print("Something Wrong")


def imgshow1(data1, data2, label, mask, idx, imgid):
    img1 = data1[0, 0, :, :, idx]
    img2 = data2[0, 0, :, :, idx]
    img3 = label[0, 0, :, :, idx]
    img4 = mask[0, 0, :, :, idx]

    fig, ax = plt.subplots(1, 4)
    img1 = img1.numpy()
    img2 = img2.numpy()
    img3 = img3.numpy()
    img4 = img4.numpy()
    ax[0].imshow(img1, cmap="gray")
    ax[0].set_title("DWI")
    ax[0].axis('off')
    ax[1].imshow(img2, cmap="gray")
    ax[1].set_title("T2")
    ax[1].axis('off')
    ax[2].imshow(img3, cmap="gray")
    ax[2].set_title("Mask")
    ax[2].axis('off')
    ax[3].imshow(img4, cmap="gray")
    ax[3].set_title("Label")
    ax[3].axis('off')
    tem = f"image2/showimg_{imgid}.png"
    plt.savefig(tem, dpi=1200, bbox_inches='tight')
    plt.show()


def imgshow2(data1, data2, label, mask, idx, imgid):
    img1 = data1[0, 0, :, :, idx]
    img2 = data2[0, 0, :, :, idx]
    # print(label.shape,mask.shape)
    img3 = label[0, 0, :, :, idx]
    img4 = mask[0, 2, :, :, idx]

    fig, ax = plt.subplots(1, 4)
    # img1 = img1.numpy()
    img2 = img2.numpy()
    img3 = img3.numpy()
    img4 = img4.numpy()
    ax[0].imshow(img1, cmap="gray")
    # ax[0].set_title("DWI")
    ax[0].axis('off')
    ax[1].imshow(img2, cmap="gray")
    # ax[1].set_title("T2")
    ax[1].axis('off')
    ax[2].imshow(img3, cmap="gray")
    ax[2].axis('off')
    ax[3].imshow(img4, cmap="gray")
    ax[3].axis('off')
    tem = f"image10/showimg_{imgid}.png"
    plt.savefig(tem, dpi=1200, bbox_inches='tight')
    plt.show()
