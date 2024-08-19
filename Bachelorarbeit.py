import MiniCode
import PlotCode
import matplotlib.pyplot as plt
import numpy as np
import Initiations
import cv2
import os
import time


def Bachelorarbeit(interesting, save=None):
    #  Initialisation for different noise parameters and different images

    res_x = 250
    res_y = 250
    # image = 'test.png'
    image = "Photographer.jpg"
    path = None
    if save is not None:
        if image == 'test.png':
            path = 'C:\Outcome\Chessboard'
        if image == 'Photographer.jpg':
            path = 'C:\Outcome\Photographer'

    kernel_size = 5
    im_matrix, im_matrix_flat, A = Initiations.Initialization_Generator(res_x, res_y, kernel_size, image)

    pre_sigma = 0.5
    pre_eta = 1
    post_sigma = 0.5
    post_eta = 1
    post_whno, pre_whno, f = Initiations.Initialization_Noise(im_matrix_flat, pre_sigma, pre_eta,
                                                              post_sigma, post_eta, A, im_matrix, path=path)

    methods = ['l1_FPC', 'l1_approx', 'l2_CG', 'l2_Landweber_step1', 'l1_Bregman_FPC', 'l1_Bregman_l1_approx',
               'l2_Bregman_CG', 'l2_Bregman_Landweber']
    nice_names_methods = ['FPC', 'l1-~', 'CG', 'LW', 'Br FPC', 'Br l1-~', 'Br CG', 'Br LW']
    images = ['Photographer.jpg', 'test.png']
    noise_level = np.linalg.norm(im_matrix_flat - f)
    noise_level_eta = 1

    alpha = 0.05
    relax = 1
    tol = 0

    fig, ax = plt.subplots(1, 1)

    itera_mini_arr = [40, 10, 10, 120]
    itera_br_arr = [3, 12, 12]

    MOI = [3]
    error_mode = "SNR"
    norm = None
    method = 'l1_Bregman_FPC'

    print(np.linalg.norm(f - im_matrix_flat))

    alphas = [0.005, 0.1, 0.2, 0.8]

    pre_whno = None
    post_whno = None
    A = np.identity(f.shape[0])

    if interesting == 'normal':
        result, residual, x_zeros, t, test = Initiations.inverse_problem(A, f, alpha, relax, itera_mini_arr[2],
                                                                   itera_br_arr[2],
                                                                   pre_whno,
                                                                   post_whno, method,
                                                                   im_matrix, noise_level,
                                                                   noise_level_eta,
                                                                   im_matrix_flat, tol, save=save,
                                                                   path=path)
        error, optimal = PlotCode.myerror(im_matrix, test, error_mode, method, norm, f)

        k = np.linspace(0, len(error) - 1, len(error))
        ax.plot(k, error, label=image)
        ax.set_xlabel(r'$kth - Iteration$')
        ax.set_ylabel('SNR - error')
        ax.legend()
        plt.xscale('log')
        plt.savefig('normal')
        plt.show(block=False)
        plt.waitforbuttonpress(0)
        plt.close('all')

    if interesting == 'ComparisonOfBreg':
        for k in range(len(MOI)):
            print(k / len(MOI))
            if MOI[k] < 0:
                result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alpha, relax, itera_mini_arr[0],
                                                                           itera_br_arr[-1],
                                                                           pre_whno,
                                                                           post_whno, methods[MOI[k]]
                                                                           , im_matrix, noise_level,
                                                                           noise_level_eta,
                                                                           im_matrix_flat, tol, save=save,
                                                                           path=path)

                error, optimal = PlotCode.myerror(im_matrix, result, error_mode, methods[MOI[k]], norm)
                k = np.linspace(0, itera_mini_arr[0] * itera_br_arr[-1], len(error))
                ax.plot(k, error, label='Bregman ' + image, linestyle='dashed', marker='x', markerfacecolor='black')
            else:
                result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alpha, relax, itera_mini_arr[-1],
                                                                           itera_br_arr[0],
                                                                           pre_whno,
                                                                           post_whno, methods[MOI[k]]
                                                                           , im_matrix, noise_level,
                                                                           noise_level_eta, save=save, path=path)

                error, optimal = PlotCode.myerror(im_matrix, result, error_mode, methods[MOI[k]], norm)
                k = np.linspace(0, len(error) - 1, len(error))
                ax.plot(k, error, label=image)

        ax.set_xlabel(r'$kth - Iteration$')
        ax.set_ylabel('SNR - error')
        ax.legend()
        plt.savefig('ComparisonOfBreg_' + str(itera_mini_arr[-1]))
        plt.show(block=False)
        plt.waitforbuttonpress(0)
        plt.close('all')

    if interesting == 'ComparisonOfImages':
        for h in range(0, len(images)):
            print(h / len(images))
            for k in range(len(MOI)):
                if MOI[k] < 0:
                    result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alpha, relax, itera_mini_arr[0],
                                                                               itera_br_arr[-1],
                                                                               pre_whno,
                                                                               post_whno, methods[MOI[k]]
                                                                               , im_matrix, noise_level,
                                                                               noise_level_eta,
                                                                               im_matrix_flat, tol, save=save,
                                                                               path=path)
                    error, optimal = PlotCode.myerror(im_matrix, result, error_mode, methods[MOI[k]], norm)
                    k = np.linspace(0, itera_mini_arr[0] * itera_br_arr[-1], len(error))
                    ax.plot(k, error, label='Bregman ' + images[h], linestyle='dashed')

                else:
                    result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alpha, relax, itera_mini_arr[-1],
                                                                               itera_br_arr[0],
                                                                               pre_whno,
                                                                               post_whno, methods[MOI[k]]
                                                                               , im_matrix, noise_level,
                                                                               noise_level_eta, save=save, path=path)
                    error, optimal = PlotCode.myerror(im_matrix, result, error_mode, methods[MOI[k]], norm)
                    k = np.linspace(0, len(error) - 1, len(error))
                    ax.plot(k, error, label=images[h])

                error, optimal = PlotCode.myerror(im_matrix, result, error_mode, methods[k], norm)
                k = np.linspace(0, len(error) - 1, len(error))
                ax.plot(k, error, label='Error of ' + images[h])

        ax.set_xlabel(r'$kth - Iteration$')
        ax.set_ylabel('SNR - error')
        ax.legend()
        plt.savefig('ComparisonOfImages_' + str(itera_mini_arr[-1]))
        plt.show(block=False)
        plt.waitforbuttonpress(0)
        plt.close('all')

    if interesting == 'ComparisonOfResults':
        color = ['b', 'r', 'g', 'm']
        for h in range(0, len(methods)):
            print(h / len(methods))
            if h > 3:
                print(methods[h])
                if methods[h] == "l2_Bregman_CG":
                    result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alpha, relax, itera_mini_arr[0],
                                                                               itera_br_arr[0],
                                                                               pre_whno,
                                                                               post_whno, methods[h]
                                                                               , im_matrix, noise_level,
                                                                               noise_level_eta,
                                                                               im_matrix_flat, save=save,
                                                                               path=path)
                if methods[h] == "l2_Bregman_Landweber":
                    result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alpha, relax, itera_mini_arr[1],
                                                                               itera_br_arr[1],
                                                                               pre_whno,
                                                                               post_whno, methods[h]
                                                                               , im_matrix, noise_level,
                                                                               noise_level_eta,
                                                                               im_matrix_flat, save=save,
                                                                               path=path)
                if methods[h] == "l1_Bregman_l1_approx" or methods[h] == "l1_Bregman_FPC":
                    result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alpha, relax, itera_mini_arr[2],
                                                                               itera_br_arr[2],
                                                                               pre_whno,
                                                                               post_whno, methods[h]
                                                                               , im_matrix, noise_level,
                                                                               noise_level_eta,
                                                                               im_matrix_flat, save=save,
                                                                               path=path)
                error, optimal = PlotCode.myerror(im_matrix, result, error_mode, methods[h], norm, f=f)
                k = np.linspace(0, itera_mini_arr[-1], len(error))
                ax.plot(k, error, label=nice_names_methods[h], linestyle='dashed', marker='o', markersize=2, color=color[h % 4])

            else:
                print(methods[h])
                result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alpha, relax, itera_mini_arr[-1],
                                                                           itera_br_arr[0],
                                                                           pre_whno,
                                                                           post_whno, methods[h]
                                                                           , im_matrix, noise_level,
                                                                           noise_level_eta, save=save, path=path)
                error, optimal = PlotCode.myerror(im_matrix, result, error_mode, methods[h], norm, f=f)
                k = np.linspace(0, len(error) - 1, len(error))
                ax.plot(k, error, label=nice_names_methods[h], color=color[h % 4])

        ax.set_xlabel(r'$kth - Iteration$')
        ax.set_ylabel('SNR - error')
        plt.xscale('log')
        ax.legend()
        plt.savefig('ComparisonOfResults_' + str(itera_mini_arr[-1]))


    if interesting == 'bias':
        pre_sigma = 1
        pre_eta = 2
        post_sigma = 1
        post_eta = 2
        post_whno, pre_whno, f = Initiations.Initialization_Noise(im_matrix_flat, pre_sigma, pre_eta,
                                                                  post_sigma, post_eta, A, im_matrix, path=path)
        pre_whno = None
        post_whno = None
        color = ['b', 'r', 'g', 'm', 'c', 'k', 'y', 'darkorange']
        for i in range(len(alphas)):
            print(i / len(alphas))
            result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alpha, relax, itera_mini_arr[-1],
                                                                       itera_br_arr[0],
                                                                       pre_whno,
                                                                       post_whno, "l2_CG"
                                                                       , im_matrix, noise_level,
                                                                       noise_level_eta,
                                                                       im_matrix_flat, save=save,
                                                                       path=path)
            error, optimal = PlotCode.myerror(im_matrix, result, error_mode, "l2_CG", norm, f)
            k = np.linspace(0, len(error) - 1, len(error))
            ax.plot(k, error, label='α =' + str(alphas[i]), color=color[i])
            result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alphas[i], relax, itera_mini_arr[0],
                                                                       itera_br_arr[0],
                                                                       pre_whno,
                                                                       post_whno, 'l2_Bregman_CG'
                                                                       , im_matrix, noise_level,
                                                                       noise_level_eta,
                                                                       im_matrix_flat, save=save,
                                                                       path=path)
            cv2.imwrite(
                'l2_Bregman_CG_' + str(alphas[i]) + '.jpg',
                result[len(result) - 1].reshape(im_matrix.shape, order='C'))
            error, optimal = PlotCode.myerror(im_matrix, result, error_mode, "l2_Bregman_CG", norm, f)
            k = np.linspace(0, len(error) - 1, len(error))
            ax.plot(k, error, label='α =' + str(alphas[i]), linestyle='dashed', marker='o', markersize=3, color=color[i])
        ax.set_xlabel(r'$kth - Iteration$')
        ax.set_ylabel('SNR - error')

        ax.legend()
        plt.xscale('log')
        plt.savefig('bias_CG')

    if interesting == 'FPC':
        iterations = [30, 50]
        bregman = [5, 3]
        for i in range(len(iterations)):
            result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alpha, relax, iterations[i],
                                                                       bregman[i],
                                                                       pre_whno,
                                                                       post_whno, 'l1_Bregman_FPC'
                                                                       , im_matrix, noise_level,
                                                                       noise_level_eta,
                                                                       im_matrix_flat, save=save,
                                                                       path=path)
            error, optimal = PlotCode.myerror(im_matrix, result, error_mode, method, norm, f)
            k = np.linspace(0, len(error) - 1, len(error))
            ax.plot(k, error, label='l1 FPC' + str(iterations[i]))
        ax.set_xlabel(r'$kth - Iteration$')
        ax.set_ylabel('SNR - error')
        ax.legend()
        plt.savefig('l1_Bregman_FPC')
        plt.show(block=False)
        plt.waitforbuttonpress(0)
        plt.close('all')

    if interesting == "ComparisonOfNoise":
        k_s = [5]
        sigma = [0.5, 1]
        eta = [1, 2]
        color = ['b', 'r', 'g', 'm', 'c', 'k', 'y', 'darkorange']
        i = 0
        for h in range(0, len(k_s)):
            kernel_size = k_s[h]
            im_matrix, im_matrix_flat, A = Initiations.Initialization_Generator(res_x, res_y, kernel_size, image)
            for j in range(len(sigma)):
                pre_sigma = sigma[j]
                post_sigma = sigma[j]
                for p in range(len(eta)):
                    pre_eta = eta[p]
                    post_eta = eta[p]
                    post_whno, pre_whno, f = Initiations.Initialization_Noise(im_matrix_flat, pre_sigma, pre_eta,
                                                                              post_sigma, post_eta, A, im_matrix,
                                                                              path=path)
                    pre_whno = None
                    post_whno = None
                    print(str(k_s[h]) + ' and ' + str(sigma[j]) + ' and ' + str(eta[p]),
                          int(np.linalg.norm(f - im_matrix_flat)))

                    result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alpha, relax, itera_mini_arr[-1],
                                                                               itera_br_arr[0],
                                                                               pre_whno,
                                                                               post_whno, "l2_CG"
                                                                               , im_matrix, noise_level,
                                                                               noise_level_eta,
                                                                               im_matrix_flat, save=save,
                                                                               path=path)
                    error, optimal = PlotCode.myerror(im_matrix, result, error_mode, "l2_CG", norm, f)
                    k = np.linspace(0, itera_mini_arr[-1], len(error))
                    ax.plot(k, error, label='δ =' + str(int(np.linalg.norm(f - im_matrix_flat))) + ", σ = " + str(sigma[j]) + ', η = ' + str(eta[p]), color=color[i])
                    if j == 0 and p == 0:
                        result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alpha, relax, 40,
                                                                                   3,
                                                                                   pre_whno,
                                                                                   post_whno, "l2_Bregman_CG"
                                                                                   , im_matrix, noise_level,
                                                                                   noise_level_eta,
                                                                                   im_matrix_flat, save=save,
                                                                                   path=path)
                        error, optimal = PlotCode.myerror(im_matrix, result, error_mode, "l2_Bregman_CG", norm, f)
                        k = np.linspace(0, itera_mini_arr[-1], len(error))
                        ax.plot(k, error, linestyle='dashed', marker='o', markersize=2, color=color[i])
                    elif j == 0 and p == 1:
                        result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alpha, relax,
                                                                                   30,
                                                                                   4,
                                                                                   pre_whno,
                                                                                   post_whno, "l2_Bregman_CG"
                                                                                   , im_matrix, noise_level,
                                                                                   noise_level_eta,
                                                                                   im_matrix_flat, save=save,
                                                                                   path=path)
                        error, optimal = PlotCode.myerror(im_matrix, result, error_mode, "l2_Bregman_CG", norm, f)
                        k = np.linspace(0, itera_mini_arr[-1], len(error))
                        ax.plot(k, error, linestyle='dashed', marker='o', markersize=2, color=color[i])
                    elif j == 1 and p == 0:
                        result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alpha, relax,
                                                                                   20,
                                                                                   6,
                                                                                   pre_whno,
                                                                                   post_whno, "l2_Bregman_CG"
                                                                                   , im_matrix, noise_level,
                                                                                   noise_level_eta,
                                                                                   im_matrix_flat, save=save,
                                                                                   path=path)
                        error, optimal = PlotCode.myerror(im_matrix, result, error_mode, "l2_Bregman_CG", norm, f)
                        k = np.linspace(0, itera_mini_arr[-1], len(error))
                        ax.plot(k, error, linestyle='dashed', marker='o', markersize=2, color=color[i])
                    elif j == 1 and p == 1:
                        result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alpha, relax,
                                                                                   10,
                                                                                   12,
                                                                                   pre_whno,
                                                                                   post_whno, "l2_Bregman_CG"
                                                                                   , im_matrix, noise_level,
                                                                                   noise_level_eta,
                                                                                   im_matrix_flat, save=save,
                                                                                   path=path)
                        error, optimal = PlotCode.myerror(im_matrix, result, error_mode, "l2_Bregman_CG", norm, f)
                        k = np.linspace(0, itera_mini_arr[-1], len(error))
                        ax.plot(k, error, linestyle='dashed', marker='o', markersize=2, color=color[i])
                    i += 1

        ax.set_xlabel(r'$kth - Iteration$')
        ax.set_ylabel('SNR - error')
        ax.legend()
        plt.xscale('log')
        plt.savefig('ComparisonOfNoise_' + str(itera_mini_arr[-1]))

#  Edge detector should give us some nice insight between l1 and l2
def edge_detector(result, image, method):
    result = result.reshape(image.shape, order='c')
    outline = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])
    result = cv2.filter2D(result, -1, outline)
    image = cv2.filter2D(image, -1, outline)

    interesting = image - result
    cv2.imshow('image', image)
    cv2.imwrite('true.jpg', image)
    cv2.imshow('image1', result)
    cv2.imwrite('edge_' + method + '.jpg', result)
    cv2.imshow('image2', interesting)
    print('Die Norm:', np.linalg.norm(interesting, 1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return
