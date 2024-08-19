import matplotlib.pyplot as plt
import numpy as np

def Lotka_Voltera():
    I = np.zeros(500)
    B = np.zeros(500)
    I[0] = 20
    B[0] = 10
    for i in range(0, len(I) - 1):
        # if i > 60 and i < 150:
        # I[i+1] = I[i] + 0.2*I[i] - 0.02*I[i]*B[i]
        # B[i+1] = B[i] + 0.002*I[i]*B[i] - 0.1*B*[i]
        # else:
        I[i + 1] = I[i] + 0.2 * I[i] - 0.01 * I[i] * B[i]
        B[i + 1] = B[i] - 0.1 * B[i] + 0.001 * I[i]

    plt.plot(np.arange(len(I)), I, label="prey")
    plt.plot(np.arange(len(B)), B, label="predator")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    Lotka_Voltera()