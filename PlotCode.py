from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import Generator
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
import NoiseCode
import main


# Plotting Images of blurstages

def blur_plot(image, blurmode, ck, whitenoisey=None, whitenoisemode=None, wnmin=None, wnmax=None, plot=None):
    Generator.image_generator("line", 100, 100, (255, 255, 255), 10)

    # whitenoisey == 1 für whitenoise auf das Bild

    if whitenoisey is not None:
        fig, ax = plt.subplots(1, 6, figsize=(10, 10))
        fig.subplots_adjust(top=0.985, bottom=0.015, left=0.015, right=0.985, hspace=0.2, wspace=0.082)

        for i in range(len(ax)):
            ax[i].axis('off')

        whnoresult, whno = NoiseCode.whitenoise(image, whitenoisemode, wnmin, wnmax, ck)  # whitenoise erstellen
        cv2.imwrite("whnoresult.jpg", whnoresult)
        result = NoiseCode.blur("whnoresult.jpg", blurmode)  # whitenoisebild blurren
        vergleich = NoiseCode.blur("linienvergleich.png", blurmode)  # Linie für Blurvergleich
        vergleich = cv2.resize(vergleich, (500, 500))
        cv2.imwrite(blurmode + "result.jpg", result)
        blurmode_temp = "Linie-" + blurmode
        cv2.imwrite("whnoresult.jpg", whnoresult)
        cv2.imwrite("whitenoise.jpg", whno)
        cv2.imwrite(blurmode_temp + ".jpg", vergleich)
        convolutionkernel = plotkernel(ck, ck.shape[0])

        ax[0].imshow(plt.imread(image), cmap='gray')
        ax[0].set_title('Original image')

        ax[1].imshow(plt.imread("whitenoise.jpg"), cmap='gray')
        ax[1].set_title('Whitenoise')

        ax[2].imshow(plt.imread("whnoresult.jpg"), cmap='gray')
        ax[2].set_title("Whitenoise result")

        ax[3].imshow(plt.imread(blurmode + "result.jpg"), cmap='gray')
        ax[3].set_title('Blurring result')

        ax[4].imshow(plt.imread(blurmode_temp + ".jpg"), cmap='gray')
        ax[4].set_title('Line comparison')

        ax[5].imshow(plt.imread(convolutionkernel))
        ax[5].set_title('Convolution kernel')

        # plt.show()

        return image, "whnoresult.jpg"

    if whitenoisey is None:
        fig, ax = plt.subplots(1, 4, figsize=(10, 10))
        fig.subplots_adjust(top=0.985, bottom=0.015, left=0.015, right=0.985, hspace=0.2, wspace=0.082)

        for i in range(len(ax)):
            ax[i].axis('off')

        result = NoiseCode.blur(image, blurmode)  # Blurring
        vergleich = NoiseCode.blur("linienvergleich.png", blurmode)  # Linie für Blurvergleich
        vergleich = cv2.resize(vergleich, (500, 500))
        cv2.imwrite(blurmode + "result.jpg", result)
        blurmode_temp = "Linie-" + blurmode
        cv2.imwrite(blurmode + ".jpg", vergleich)
        convolutionkernel = plotkernel(ck, ck.shape[0])

        ax[0].imshow(plt.imread(image), cmap='gray')
        ax[0].set_title('Original image')

        ax[1].imshow(plt.imread(blurmode + "result.jpg"), cmap='gray')
        ax[1].set_title(blurmode + 'noise')

        ax[2].imshow(plt.imread(blurmode_temp + ".jpg"), cmap='gray')
        ax[2].set_title('line comparison')

        ax[3].imshow(plt.imread(convolutionkernel))
        ax[3].set_title('Convolution Kernel')

        # plt.show()

        return image, blurmode + "result.jpg"


"""
image: generiertes Image oder implementiertes Image
blurmode: (convolution)kernel oder gaussianblur
kernelmode: gaussian3, gaussian5, boxblur, edge0, edge1, edge2, sharpen oder emboss
whitenosey: 0 for no whitenoise, 1 for whitenoise
whitenoisemode: specklenoise oder gaussian
wnmin/wnmax: 0 <= wnmin < wnmax <= 1
"""


# Plotting Error

def myerror(true_image, Images_Array, error_mode, method, norm=None, f = None):
    img_m = true_image.flatten(order='C')

    if f is None:
        img_fil_m = Images_Array[0, :]
    else:
        img_fil_m = f
    relative = np.linalg.norm(img_m - img_fil_m, ord=norm)
    error = []

    for k in range(0, len(Images_Array)):
        img_del_fil_m = Images_Array[k, :]

        if error_mode == "absolute":

            error += [np.linalg.norm(img_m - img_del_fil_m, ord=norm)]

        elif error_mode == "relative":

            error += [np.linalg.norm(img_m - img_del_fil_m, ord=norm) / relative]

        elif error_mode == "scaled" or error_mode == "SNR":

            error += [np.linalg.norm(img_m - img_del_fil_m) / np.linalg.norm(f)]

            if error_mode == "SNR":
                error[k] = - math.log(error[k])


        else:
            print("error_mode: absolute, relative, scaled, SNR")

        """if data_fidelity_term[k] > data_fidelity_term[k-1] and k > 0:
            array_length = k + 1
            error = error[0:k + 1]
            bias = bias[0:k + 1]
            variance = variance[0:k + 1]
            data_fidelity_term = data_fidelity_term[0:k + 1]
            break"""

    """ax_result[0].imshow(true_image, cmap='gray')
    ax_result[0].set_title('True Image')

    ax_result[1].imshow(Images_Array[0, :].reshape(true_image.shape, order='C'), cmap='gray')
    ax_result[1].set_title('Corrupted Image')"""

    for i in range(1, len(Images_Array) - 1):
        if error[i] > error[i + 1]:
            cv2.imwrite(method + '_optimal_' + str(i) + '.jpg',
                        Images_Array[i, :].reshape(true_image.shape, order='C'))
            optimal = i
            break
        else:
            optimal = i + 1
            continue

    """for i in range(0, len(Images_Array)):
        step = Images_Array[i, :].reshape(true_image.shape, order='C')
        ax_result[i + 2].imshow(step, cmap='gray')
        ax_result[i + 2].set_title('Step' + str(i))

    for i in range(len(ax_result)):
        ax_result[i].axis('off')"""

    # print(method)
    # print("Error:", error)

    return error, optimal

def residual_plot(residual):
    # print('Residual:', residual)
    length = len(residual)
    k = np.linspace(1, length, length)
    fig, ax = plt.subplots(1, 1)

    ax.plot(k, residual, label='Residual')
    ax.set_xlabel(r'iteration')
    ax.set_ylabel(r'Residual in iteration')
    ax.legend()

    plt.show(block=False)
    plt.waitforbuttonpress(0)
    plt.close('all')



def Bias_plot(A, f, alpha, relax, itera_mini, itera_br, pre_whno, post_whno, method, step,
              iterations, im_matrix, norm=None):

    bias_array = []
    for i in range(0, iterations):
        u_hat, residual = main.inverse_problem(A, f, 0, relax, itera_mini, itera_br, pre_whno, post_whno, method,
                                                step, im_matrix)
        if itera_br > 0:
            u_hat = u_hat[itera_br - 1]
        u_hat = u_hat[itera_mini - 1]
        f_alpha_hat, residual = main.inverse_problem(A, f, alpha * i/iterations, relax, itera_mini, itera_br, pre_whno,
                                                     post_whno, method, step, im_matrix)
        if itera_br > 0:
            f_alpha_hat = f_alpha_hat[itera_br - 1]
        f_alpha_hat = f_alpha_hat[itera_mini - 1]
        bias = np.linalg.norm(u_hat - f_alpha_hat, ord=norm)
        if bias != 0:
            bias_array += [math.log(bias)]
        else:
            bias_array += [0]
    print('Bias:', bias_array)
    k = np.linspace(0, alpha, iterations)
    fig, ax = plt.subplots(1, 1)
    ax.plot(k, bias_array, label='Bias')
    ax.set_xlabel(r'alpha stability term')
    ax.set_ylabel(r'Logarithmic scaled')
    ax.legend()

    plt.show(block=False)
    plt.waitforbuttonpress(0)
    plt.close('all')


# Plotting convolutionkernel

def plotkernel(mykernel, ksize):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X = np.arange(0, ksize, 1)
    Y = np.arange(0, ksize, 1)
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, mykernel, rstride=1,
                           cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(0, np.max(mykernel))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('convolutionkernel.png')
    return 'convolutionkernel.png'
