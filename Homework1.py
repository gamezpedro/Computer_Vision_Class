import numpy as np

def list_exercices():
    colors = ["blue", "white", "green", "red", "pink", "yellow"]
    colors.extend(["violet", "grey"])
    colors
    colors[0] = "cyan"
    del colors[1:3]
    colors

def dictionary_exercices():
    dictionary = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'}
    dictionary[6] = 'f'
    del dictionary[2]
    dictionary

def numpy_exercices():
    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    sampArray = np.array([[11 ,22, 33], [44, 55, 66], [77, 88, 99]])
    outputArray = (sampArray[0,2], sampArray[1,2], sampArray[2,2])
    outputArray
    arr = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37])
    arr[::-1]

if __name__ == "__main__":
    list_exercices
    dictionary_exercices
    numpy_exercices