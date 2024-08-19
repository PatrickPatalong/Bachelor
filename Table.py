import numpy as np
import cv2
from tabulate import tabulate
import Initiations


def Table(interesting, norm=None):
    #  Initialisation for different noise parameters and different images

    res_x = 250
    res_y = 250
    # image = 'test.png'
    image = "Photographer.jpg"
    kernel_size = 5
    im_matrix, im_matrix_flat, A = Initiations.Initialization_Generator(res_x, res_y, kernel_size, image)

    pre_sigma = 0.5
    pre_eta = 1
    post_sigma = 0.5
    post_eta = 1
    post_whno, pre_whno, f = Initiations.Initialization_Noise(im_matrix_flat, pre_sigma, pre_eta,
                                                              post_sigma, post_eta, A, im_matrix)
    pre_whno = None
    post_whno = None
    #  All the methods that will be compared
    methods = ['l1_FPC', 'l1_approx', 'l2_CG', 'l2_Landweber_step1', 'l1_Bregman_FPC', 'l1_Bregman_l1_approx',
               'l2_Bregman_CG', 'l2_Bregman_Landweber']
    nice_names_methods = ['FPC', 'l_1-~', 'CG', 'LW', 'Breg_FPC', 'Breg_l_1-~', 'Breg_CG', 'Breg_LW']
    table = []
    #  iterate through all methods with same parametes

    if interesting == "TimeError":
        skip = []
        tol = [0.9, 0.7, 0.5, 0.45, 0.4, 0.35]
        alpha = 0.05
        itera_mini_arr = [20, 5, 10, 600]
        itera_br_arr = [30, 120, 60]
        relax = 1
        itera_br = 1
        relative = np.linalg.norm( im_matrix_flat - f)
        result_CG, residual_CG, x_zeros_CG, t_CG = Initiations.inverse_problem(A, f, alpha, relax, itera_mini_arr[-1],
                                                                               itera_br,
                                                                               pre_whno,
                                                                               post_whno, "l2_CG", im_matrix,
                                                                               true=im_matrix_flat,
                                                                               tol=0)
        p = 1
        while np.linalg.norm(im_matrix_flat - result_CG[p]) / relative <= np.linalg.norm(im_matrix_flat - result_CG[p-1]) / relative :
            p += 1
        print(np.linalg.norm(im_matrix_flat - result_CG[p]) / relative)

        for j in range(len(tol)):
            table_temp = [tol[j]]
            for i in range(len(methods)):
                method = methods[i]
                print(method)
                if method in skip:
                    table_temp += ['NaN']
                    table += [table_temp]
                    continue
                alpha = 0.05
                if i < 4:
                    itera_mini = itera_mini_arr[-1]
                    itera_br = 25
                else:
                    if method == "l2_Bregman_CG":
                        itera_mini = itera_mini_arr[0]
                        itera_br = itera_br_arr[0]
                    if method == "l2_Bregman_Landweber":
                        itera_mini = itera_mini_arr[1]
                        itera_br = itera_br_arr[1]
                    if method == "l1_Bregman_l1_approx" or method == "l1_Bregman_FPC":
                        itera_mini = itera_mini_arr[2]
                        itera_br = itera_br_arr[2]
                relax = 1

                if method == "l2_CG":
                    if np.linalg.norm(im_matrix_flat - result_CG[p]) / relative < tol[j]:
                        k = 1
                        while np.linalg.norm(im_matrix_flat - result_CG[k]) / relative > tol[j]:
                            k += 1
                        result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alpha, relax, k, itera_br,
                                                                                   pre_whno,
                                                                                   post_whno, method, im_matrix,
                                                                                   true=im_matrix_flat,
                                                                                   tol=tol[j])
                        table_temp += [round(t, 2)]
                    else:
                        print(str(np.linalg.norm(im_matrix_flat - result_CG[-1]) / relative) + ">" + str(tol[j]))
                        table_temp += ['NaN']
                        skip += [method]

                else:
                    result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alpha, relax, itera_mini, itera_br,
                                                                               pre_whno,
                                                                               post_whno, method, im_matrix,
                                                                               true=im_matrix_flat,
                                                                               tol=tol[j])

                    if np.linalg.norm(im_matrix_flat - result[-1]) / relative > tol[j]:
                        print(str(np.linalg.norm(im_matrix_flat - result[-1]) / relative) + ">" + str(tol[j]))
                        table_temp += ['NaN']
                        skip += [method]
                        print(skip)
                    else:
                        table_temp += [round(t, 2)]
            table += [table_temp]
        headers = ["Error"] + nice_names_methods
        np.array(table, dtype=object)

    print(tabulate(table, headers, tablefmt="latex"))


    if interesting == "normal":
        for i in range(len(methods)):
            method = methods[i]
            nice_method = nice_names_methods[i]
            alpha = 0.05
            itera_mini = 5
            relax = 1
            itera_br = 3
            result, residual, x_zeros, t = Initiations.inverse_problem(A, f, alpha, relax, itera_mini, itera_br,
                                                                       pre_whno,
                                                                       post_whno, method, im_matrix)

            error = np.linalg.norm(im_matrix_flat - result[len(result) - 1, :], norm) \
                    / np.linalg.norm(im_matrix_flat - result[0], norm)
            table += [[nice_method, round(t, 2), round(residual[len(residual) - 1], 2), error]]

        np.array(table, dtype=object)
        headers = ["Method", "Time", "Residual", "Error"]
        print(table, headers)



