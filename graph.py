import matplotlib.pyplot as plt


def plot(matrix, initial, number_components, accuracy):
    # plotting accuracy against number of gaussian components
    if matrix == 'diag' and initial == 'kmeans':
        output = 'DIAG_KMEANS.png'
        f1 = plt.figure()
        plt.plot(number_components, accuracy)
        plt.title('EFFECT OF NUMBER OF GAUSSIAN COMPONENTS ON ACCURACY with diag and kmeans')
        plt.ylabel('ACCURACY')
        plt.xlabel('NUMBER_GAUSSIAN_COMPONENT')
        plt.show()
        f1.savefig(output)
    if matrix == 'full' and initial == 'kmeans':
        output = 'FULL_KMEANS.png'
        f2 = plt.figure()
        plt.plot(number_components, accuracy)
        plt.title('EFFECT OF NUMBER OF GAUSSIAN COMPONENTS ON ACCURACY full and kmeans')
        plt.ylabel('ACCURACY')
        plt.xlabel('NUMBER_GAUSSIAN_COMPONENT')
        plt.show()
        f2.savefig(output)
    if matrix == 'tied' and initial == 'kmeans':
        output = 'TIED_KMEANS.png'
        f3 = plt.figure()
        plt.plot(number_components, accuracy)
        plt.title('EFFECT OF NUMBER OF GAUSSIAN COMPONENTS ON ACCURACY tied and kmeans')
        plt.ylabel('ACCURACY')
        plt.xlabel('NUMBER_GAUSSIAN_COMPONENT')
        plt.show()
        f3.savefig(output)
    if matrix == 'diag' and initial == 'random':
        output = 'DIAG_RANDOM.png'
        f4 = plt.figure()
        plt.plot(number_components, accuracy)
        plt.title('EFFECT OF NUMBER OF GAUSSIAN COMPONENTS ON ACCURACY diag and random')
        plt.ylabel('ACCURACY')
        plt.xlabel('NUMBER_GAUSSIAN_COMPONENT')
        plt.show()
        f4.savefig(output)
    if matrix == 'full' and initial == 'random':
        output = 'FULL_RANDOM.png'
        f5 = plt.figure()
        plt.plot(number_components, accuracy)
        plt.title('EFFECT OF NUMBER OF GAUSSIAN COMPONENTS ON ACCURACY')
        plt.ylabel('ACCURACY')
        plt.xlabel('NUMBER_GAUSSIAN_COMPONENT')
        plt.show()
        f5.savefig(output)
    if matrix == 'tied' and initial == 'random':
        output = 'TIED_RANDOM.png'
        f6 = plt.figure()
        plt.plot(number_components, accuracy)
        plt.title('EFFECT OF NUMBER OF GAUSSIAN COMPONENTS ON ACCURACY tied and random')
        plt.ylabel('ACCURACY')
        plt.xlabel('NUMBER_GAUSSIAN_COMPONENT')
        plt.show()
        f6.savefig(output)
