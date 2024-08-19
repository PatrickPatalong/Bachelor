import NoiseCode
import Generator
import numpy as np
import matplotlib.pyplot as plt

def Variance_plot(iterations, kernel_size, img_matrix, alpha, itera, relax, denoise_mode, norm=None):
    kernel_array = np.zeros(iterations)
    variance_array = np.zeros(iterations)
    if iterations > kernel_size:
        print('It is necessary that iterations < kernel_size')

    for i in range(1, iterations):
        kernel_temp = int(kernel_size / i)
        kernel_array += [kernel_temp]
        Convo = Generator.make_blur_matrix(img_matrix, kernel_temp)
        img_flat = img_matrix.flatten(order='C')
        u_del = NoiseCode.blur(img_matrix, "kernel", Convo)
        u_del_flatten = u_del.flatten(order='C')

        result = MiniCode.Landweber_iteration(u_del_flatten, Convo, alpha, itera, relax, denoise_mode)
        f_alpha_hat = MiniCode.Landweber_iteration(img_flat, Convo, alpha, itera, relax, denoise_mode)
        f_alpha_hat = f_alpha_hat[itera - 1, :]
        variance = np.linalg.norm(f_alpha_hat - result[itera - 1, :], ord=norm)
        variance_array += [variance]

    print('Variance:', variance_array)
    fig, ax = plt.subplots(1, 1)
    ax.plot(kernel_array, variance_array, label='Variance')
    ax.set_xlabel(r'blurring term')
    ax.set_ylabel(r'delta')
    ax.legend()

    plt.show(block=False)
    plt.waitforbuttonpress(0)
    plt.close('all')

def Bias_Variance_plot(iterations, img_flat, Degrad, Gam, iter, relax, denoise_mode, result, norm=None):
    alpha = []
    bias_array = []
    variance_array = []
    for i in range(0, iterations):
        u_hat = MiniCode.Landweber_iteration(img_flat, Degrad, 0, iter, relax, denoise_mode)
        u_hat = u_hat[iter - 1, :]
        f_alpha_hat = MiniCode.Landweber_iteration(img_flat, Degrad, Gam * i, iter, relax, denoise_mode)
        f_alpha_hat = f_alpha_hat[iter - 1, :]
        bias = np.linalg.norm(u_hat - f_alpha_hat, ord=norm)
        variance = np.linalg.norm(f_alpha_hat - result[iter - 1, :], ord=norm)
        alpha += [i * Gam]
        if bias != 0:
            bias_array += [math.log(bias)]
            variance_array += [math.log(variance)]
        else:
            bias_array += [0]
            variance_array += [math.log(variance)]
    print('Bias:', bias_array)
    print('Variance:', variance_array)
    fig, ax = plt.subplots(1, 1)
    ax.plot(alpha, bias_array, label='Bias')
    ax.plot(alpha, variance_array, label='Variance')
    ax.set_xlabel(r'alpha stability term')
    ax.set_ylabel(r'Logarithmic scaled')
    ax.legend()

    plt.show(block=False)
    plt.waitforbuttonpress(0)
    plt.close('all')


def randomness():
    alpha = 0.025
    itera_mini = 20
    relax = 1
    for i in range(len(etas)):
        for j in range(len(sigmas)):
            sigma = sigmas[j]
            eta = etas[i]

            result, residual, t = Landweber.Landweber(A, f, alpha, itera_mini, relax, pre_whno, post_whno, eta, sigma)

            error = np.linalg.norm(im_matrix_flat - result[len(result) - 1, :], norm) \
                    / np.linalg.norm(im_matrix_flat - result[0], norm)
            error_array[i, j] = error
            print(error_array)
            table_temp = [sigmas, etas, error]
            table = table + [table_temp]


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(etas, sigmas)
    surf = ax.plot_surface(X, Y, error_array, rstride=1,
                           cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(0, np.max(error_array))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('Randomness.png')

def test():
    res_x = 250
    res_y = 250
    # image = 'test.png'
    image = "Photographer.jpg"
    kernel_size = 5
    im_matrix, im_matrix_flat, A = main.Initialization_Generator(res_x, res_y, kernel_size, image)

    pre_sigma = 2
    pre_eta = 1
    post_sigma = 2
    post_eta = 1
    pre_noise = "gaussian"
    post_noise = "gaussian"
    post_whno, pre_whno, whno_im, post_noise, f = main.Initialization_Noise(im_matrix_flat, pre_sigma, pre_eta,
                                                                            post_sigma, post_eta, pre_noise, post_noise,
                                                                            A, im_matrix)
    u = f
    x = f
    x_temp = x
    for i in range(0, 100):
        threshhold = 1 / 2 * np.linalg.norm(A.dot(u) + post_whno - f)
        stand = threshhold
        while stand >= threshhold:
            print(i)
            x_temp = x
            for k in range(f.shape[0]):
                whno = pre_eta * random.gauss(0, pre_sigma ** 2)
                x_temp[k] = x_temp[k] + whno
            x_temp = A.dot(x_temp)
            for k in range(f.shape[0]):
                whno = pre_eta * random.gauss(0, pre_sigma ** 2)
                x_temp[k] = x[k] + whno
            stand = 1 / 2 * np.linalg.norm(x_temp - f)
        u = x_temp
    print(stand)

    cv2.imwrite("test.jpg", u.reshape(im_matrix.shape, order='C'))