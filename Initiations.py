import cv2
import matplotlib.pyplot as plt
import PlotCode
import Generator
import NoiseCode
import Landweber
import MiniCode
import numpy as np
import os
import Table


def inverse_problem(A, f, alpha, relax, itera_mini, itera_br, pre_whno, post_whno, method, im_matrix, delta=0, eta=0,
                    true=None, tol=0,
                    save=None, path=None):
    x_zeros = None
    if true is None:
        im_matrix_flat = im_matrix.flatten(order='C')
        true = np.zeros((im_matrix_flat.shape[0]))

    if method == "l2_CG":
        # CG Iteration from scipy
        MiniCode.cg_timer = 0
        MiniCode.cg_nresid = []
        MiniCode.cg_result = np.zeros((itera_mini, im_matrix.shape[0] * im_matrix.shape[1]))
        result, residual, t = MiniCode.solve_tykhonov_sparse(A, f, alpha, itera_mini, pre_whno, post_whno)
        # print("Scipy_CG method took:", t)
        if path is not None:
            path = path + '\l2_CG'
            cv2.imwrite(os.path.join(path, 'result_l2_cg_' + str(itera_mini) + '.jpg'),
                        result[itera_mini - 1].reshape(im_matrix.shape, order='C'))
        if save is not None:
            cv2.imwrite('result_l2_cg_' + str(itera_mini) + '.jpg',
                        result[itera_mini - 1].reshape(im_matrix.shape, order='C'))
        itera_mini = len(result)

    if method == "l2_Landweber_step1":
        result, residual, t, cond = Landweber.Landweber(A, f, alpha, itera_mini, relax, pre_whno, post_whno, true, tol)
        print("Conditioning of K:", cond)
        # print("Landweber with 1 step in l2 took:", t)
        if path is not None:
            path = path + '\l2_Landweber'
            cv2.imwrite(os.path.join(path, 'result_Landweber_l2_1step_' + str(itera_mini) + '.jpg'),
                        result[itera_mini - 1, :].reshape(im_matrix.shape, order='C'))
        if save is not None:
            cv2.imwrite('result_Landweber_l2_1step_' + str(itera_mini) + '.jpg',
                        result[itera_mini - 1, :].reshape(im_matrix.shape, order='C'))

    if method == "l2_Landweber_step2":
        result, residual, t1 = Landweber.Landweber_denoising(f, post_whno, alpha, itera_mini, relax)
        result, residual, t2 = Landweber.Landweber_deblurring(result[itera_mini - 1], A, alpha, itera_mini, relax)
        result, residual, t3 = Landweber.Landweber_denoising(result[itera_mini - 1], pre_whno, alpha, itera_mini, relax)
        t = t1 + t2 + t3
        # print("Landweber with 2 steps in l2 took:", t)
        if path is not None:
            path = path + '\l2_Landweber_step2'
            cv2.imwrite(os.path.join(path, 'result_Landweber_l2_2steps_' + str(itera_mini) + '.jpg'),
                        result[itera_mini - 1, :].reshape(im_matrix.shape, order='C'))
        if save is not None:
            cv2.imwrite('result_Landweber_l2_2steps_' + str(itera_mini) + '.jpg',
                        result[itera_mini - 1, :].reshape(im_matrix.shape, order='C'))

    if method == "l2_Bregman_CG":
        MiniCode.cg_timer = 0
        MiniCode.cg_nresid = []
        MiniCode.cg_result = np.zeros((itera_br * itera_mini, im_matrix.shape[0] * im_matrix.shape[1]))
        result, residual, t = MiniCode.Bregman_iteration(A, f, alpha, 2, itera_mini, itera_br, "CG",
                                                         pre_whno, post_whno, delta, eta, true, tol)
        # print("Bregman with scipy_CG in l2 took:", t)
        if path is not None:
            path = path + '\l2_Bregman_CG'
            cv2.imwrite(os.path.join(path, method + '_br_' + str(itera_br) + '_mini_' + str(itera_mini) + '.jpg'),
                        result[itera_br - 1].reshape(im_matrix.shape, order='C'))
        if save is not None:
            cv2.imwrite(method + '_br_' + str(itera_br) + '_mini_' + str(itera_mini) + '.jpg',
                        result[itera_br - 1].reshape(im_matrix.shape, order='C'))
        itera_mini = len(result)

    if method == "l2_Bregman_Landweber":
        MiniCode.cg_timer = 0
        MiniCode.cg_nresid = []
        MiniCode.cg_result = np.zeros((itera_br * itera_mini, im_matrix.shape[0] * im_matrix.shape[1]))
        result, residual, t, cond= MiniCode.Bregman_iteration(A, f, alpha, 2, itera_mini, itera_br, "Landweber",
                                                         pre_whno, post_whno, delta, eta, true, tol)
        print("Conditioning of K:", cond)
        # print("Bregman with Landweber in l2 took:", t)
        if path is not None:
            path = path + '\l2_Bregman_Landweber'
            cv2.imwrite(os.path.join(path, method + '_br_' + str(itera_br) + '_mini_' + str(itera_mini) + '.jpg'),
                        result[itera_br - 1].reshape(im_matrix.shape, order='C'))
        if save is not None:
            cv2.imwrite(method + '_br_' + str(itera_br) + '_mini_' + str(itera_mini) + '.jpg',
                        result[itera_br - 1].reshape(im_matrix.shape, order='C'))
        itera_mini = len(result)

    if method == "l1_FPC":
        result, residual, t = MiniCode.FPC(A, f, alpha, itera_mini, pre_whno, post_whno, relax, true, tol)
        # print("FPC with l1 took:", t)
        if path is not None:
            path = path + '\l1_FPC'
            cv2.imwrite(os.path.join(path, 'result_FPC_l1_deblur_' + str(itera_mini) + '.jpg'),
                        result[itera_mini - 1, :].reshape(im_matrix.shape, order='C'))
        if save is not None:
            cv2.imwrite('result_FPC_l1_deblur_' + str(itera_mini) + '.jpg',
                        result[itera_mini - 1, :].reshape(im_matrix.shape, order='C'))

    if method == "l1_approx":
        result, residual, x_zeros, t = MiniCode.l1_approx(A, f, alpha, itera_mini, pre_whno, post_whno, relax, true,
                                                          tol)
        # print("l1_approx in l1 took:", t)
        if path is not None:
            path = path + '\l1_approx'
            cv2.imwrite(os.path.join(path, 'result_l1_approx_' + str(itera_mini) + '.jpg'),
                        result[itera_mini - 1].reshape(im_matrix.shape, order='C'))
        if save is not None:
            cv2.imwrite('result_l1_approx_' + str(itera_mini) + '.jpg',
                        result[itera_mini - 1].reshape(im_matrix.shape, order='C'))

    if method == "l1_Bregman_FPC":
        result, residual, t = MiniCode.Bregman_iteration(A, f, alpha, 1, itera_mini, itera_br, "FPC",
                                                         pre_whno, post_whno, delta, eta, true, tol)
        # print("Bregman with FPC in l1 took:", t)
        if path is not None:
            path = path + '\l1_Bregman_FPC'
            cv2.imwrite(
                os.path.join(path, 'result_l1_Bregman_FPC_br_' + str(itera_br) + '_mini_' + str(itera_mini) + '.jpg'),
                result[itera_br - 1].reshape(im_matrix.shape, order='C'))
        if save is not None:
            cv2.imwrite(
                'result_l1_Bregman_FPC_br_' + str(itera_br) + '_mini_' + str(itera_mini) + '.jpg',
                result[itera_br - 1].reshape(im_matrix.shape, order='C'))

    if method == "l1_Bregman_l1_approx":
        result, residual, x_zeros, t = MiniCode.Bregman_iteration(A, f, alpha, 1, itera_mini, itera_br,
                                                                  "l1_approx", pre_whno, post_whno, delta, eta, true,
                                                                  tol)
        # print("Bregman with l1_approx in l1 took:", t)
        if path is not None:
            path = path + '\l1_Bregman_l1_approx'
            cv2.imwrite(os.path.join(path, 'result_l1_Bregman_l1_approx_br' + str(itera_br) + '_mini_' + str(
                itera_mini) + '.jpg'), result[itera_br - 1].reshape(im_matrix.shape, order='C'))
        if save is not None:
            cv2.imwrite('result_l1_Bregman_l1_approx_br' + str(itera_br) + '_mini_' + str(
                itera_mini) + '.jpg', result[len(result) - 1].reshape(im_matrix.shape, order='C'))

    return result, residual, x_zeros, t


def Initialization_Generator(res_x=250, res_y=250, kernel_size=5, image=None):
    if image == 'test.png':
        # Image initiation
        image = Generator.chess_generator(res_x, 255)
    im_matrix = cv2.imread(image, 0)
    im_matrix_flat = im_matrix.flatten(order='C')

    #  Blur parameters
    kernel_size = kernel_size
    A = Generator.make_blur_matrix(im_matrix, kernel_size)

    return im_matrix, im_matrix_flat, A


def Initialization_Noise(im_matrix_flat, pre_sigma, pre_eta, post_sigma, post_eta, A, im_matrix, pre_noise="gaussian",
                         post_noise="gaussian", path=None):
    ################################################ Pre-Noise #########################################################
    if pre_noise is None:
        pre_whno = np.zeros(im_matrix_flat.shape[0])
        f = im_matrix_flat
    else:
        f, pre_whno = NoiseCode.whitenoise(im_matrix_flat, im_matrix, pre_noise, pre_sigma, pre_eta)
        whno_im = pre_whno.reshape(im_matrix.shape, order='C')
        if path is not None:
            cv2.imwrite(os.path.join(path, 'pre_whno.jpg'), whno_im)
        cv2.imwrite('pre_whno.jpg', whno_im)

    ################################################### Blur ###########################################################

    f = NoiseCode.blur(f, A)

    ########################################0########  Post-Noise #######################################################
    if post_noise is None:
        post_whno = np.zeros(im_matrix_flat.shape[0])
    else:
        f, post_whno = NoiseCode.whitenoise(f, im_matrix, post_noise, post_sigma, post_eta)
        whno_im = post_whno.reshape(im_matrix.shape, order='C')
        if path is not None:
            cv2.imwrite(os.path.join(path, 'post_whno.jpg'), whno_im)
        cv2.imwrite('post_whno.jpg', whno_im)

    ######################################## End of Initialisation #####################################################
    if path is not None:
        cv2.imwrite(os.path.join(path, 'noised_image.jpg'), f.reshape(im_matrix.shape, order='C'))
    cv2.imwrite('noised_image.jpg', f.reshape(im_matrix.shape, order='C'))

    return post_whno, pre_whno, f
