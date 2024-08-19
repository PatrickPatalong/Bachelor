import Bachelorarbeit
import Table
import matplotlib.pyplot as plt



if __name__ == "__main__":

    # Table.Table("normal")

    ############################################## Initiation/Generator ###############################################
    """
    The first block is for image generation or initiation. You can inititate any image you have on your disk, e.g.
    res_x = 250
    res_y = 250
    image = 'test.png'
    # image = "Photographer.jpg"
    """
    ########################################### Initiation/Noise #######################################################
    """
    This block is about noise. First of all we are generating our blur matrix
    kernel_size = 5
    im_matrix, im_matrix_flat, A = Initiations.Initialization_Generator(res_x, res_y, kernel_size, image)
    
     Using no noise define pre_noise and post_noise as None. Otherwise use the parameters
    sigma and eta as noise parameters, e.g.
    
    pre_sigma = 0.5
    pre_eta = 1
    post_sigma = 0.5
    post_eta = 1
    pre_noise = "gaussian"
    post_noise = "gaussian"
    
    path = None
    if image == 'test.png':
        path = 'C:\Outcome\Chessboard'
    if image == 'Photographer.jpg':
        path = 'C:\Outcome\Photographer'

    post_whno_first, pre_whno_first, f_first = Initiations.Initialization_Noise(im_matrix_flat, pre_sigma, pre_eta, post_sigma,
                                                                    post_eta, A, im_matrix, pre_noise, post_noise, path)
    """

    ############################################### Initiation/Method ##################################################
    """
    We may use different denoising methods. All of them are listed below:
    For denoise mode = 1 (l1- regularization) we have the methods
        l1_FPC
        l1_approx (MAYBE implement abschätzung)
        l1_Bregman_FPC (Bregman iteration with the FPC method)
        l1_Bregman_l1_approx (Bregman iteration with the l1 approx method)
    For denoise mode = 2 (l2 - regularization) we have the methods
        l2_CG
        l2_Landweber_step1      (Here we may use two different approaches: 
        l2_Landweber_step2      deblurring and denoising in 1 step or in 2 steps
        l2_Bregman_CG           (Here we may differ between the two inner methods:
        l2_Bregman_Landweber    Landweber and CG)
    
    Lastly we have some parameters for our minimization problem:
        alpha - regularization parameter
        itera_cg - iteration for minimization problem
        relax - stepsize/relaxation parameter
        itera_br - Bregman iteration
    
    alpha = 0.05
    itera_mini = 5
    relax = 1  # relaxation parameter (vgl. tau in paper)
    itera_br = 5
    
    For analysis of all methods we may use the parameters:
    itera_mini_arr = [5, 25, 50, 75]
    itera_br_arr = [1, 5, 10, 15]
    methods = ['l1_FPC', 'l1_approx', 'l2_CG', 'l2_Landweber_step1', 'l1_Bregman_FPC', 'l1_Bregman_l1_approx',
               'l2_Bregman_CG', 'l2_Bregman_Landweber']
    images = ['Photographer.jpg', 'test.png']
    """
    # TODO: Discrepany principle and alpha = array
    # TODO: A : R^m -> R^n

    ########################################### Playground ############################################################
    # Table.Table('TimeError')

    Bachelorarbeit.Bachelorarbeit('ComparisonOfResults')
    plt.show(block=False)
    plt.waitforbuttonpress(0)
    plt.close('all')

    #Bachelorarbeit.Bachelorarbeit('bias')
    #plt.show(block=False)
    #plt.waitforbuttonpress(0)
    #plt.close('all')
    # Table.Table("TimeError")





    ########################################### Variance & Bias #######################################################
    # Initialisation
    # bias_itera = 5

    # Problem is, that we find different convolution matrices, dependend on the initial point
    # Bias is changing with alpha
    # IMPORTANT: IF NOT USING BREGMAN -> itera_br = 0 !!!
    # denoise_mode = int(method[1])
    # PlotCode.Bias_plot(A, f, alpha, relax, itera_mini, itera_br, pre_whno, post_whno, method, inner_method, denoise_mode
    #                   , step, bias_itera, im_matrix)

    # Variance is changing with Convo
    # Variance is not interessting right now
    # Bin.Variance_plot(Iterations, kernel_size, true_image_matrix, alpha, itera, relax, denoise_mode)