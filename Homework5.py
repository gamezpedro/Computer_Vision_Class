'''
Student: Victor Sebastian Martinez Perez
ID: A01232474
'''
import cv2
import os
import random
import matplotlib.pyplot as plt
import numpy as np

def main():
    ## PART 1 ##
    train_dir = os.path.join(os.getcwd(), "train")
    classes = os.listdir(train_dir)
    numo_images_list = []

    for i in range(len(classes)):
        class_dir = os.path.join(train_dir, classes[i])
        numo_images_list.append(len([file for file in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir,file))]))

        # Initialize the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_pos = np.arange(len(classes)) # Return evenly spaced values within a given interval (0, 1, ..., len(classes))

        # Plot the data
    ax.bar(x_pos, numo_images_list)
    plt.xticks(x_pos, classes)
    plt.ylabel("Number of images")
    plt.title("Images by class")

        # Show or save
    plt.show()
    #plt.savefig("Bar_graph.png")
    plt.close()

    ## PART 2 ##
    random_class = os.path.join(train_dir, random.choice(classes))
    random_file = os.path.join(random_class, random.choice(os.listdir(random_class)))
    image = cv2.imread(random_file)

    rotated = cv2.rotate(image, cv2.ROTATE_180)
    mirror = cv2.flip(image, 1)

    cv2.imshow("Original", image)
    cv2.imshow("Rotated", rotated)
    cv2.imshow("Mirror", mirror)

    print(f"press 'S' if you want to save the images. Press any other key if not")
    k = cv2.waitKey(0) & 0xFF
    if k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite("Rotation.png",rotated)
        cv2.imwrite("Mirror.jpg", mirror)
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()

    ## PART 3 ##
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.resize(gray_img, (300,200))
    cv2.imshow("Original", image)
    cv2.imshow("Gray+resized", gray_img)
    print(f"press 'S' if you want to save the images. Press any other key if not")
    k = cv2.waitKey(0) & 0xFF
    if k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite("Gray+resized.png",gray_img)
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()

    ## PART 4 ##
    '''
    Posiblemente se divida en 80/20 para poder entrenar el modelo con ese 80% y con
    el 20% restante, dar una evaluacion de que tan bueno fue el rendimiento del modelo
    y tal vez para darle unos ultimos ajustes
    '''
    
if __name__ == "__main__":
    main()
