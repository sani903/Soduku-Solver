# Soduku Solver
Takes in soduku images as input with handwritten digits and classifies the digits and identifies box edges.
<br>
It then solves these sodukus and reprojects the solutions on the original image.
<br>
I have trained a CNN from the MNIST database to help recognize the digits with an accuracy of 99.06%
<br>
This program succeeds in identifying the rows and columns from the sudoku puzzle and recognizing the digits.
<br>
Due to the use of a custom CNN, the accuracy is not high enough to predict every digit of the puzzle correctly.
<br>
Due to incorrect predictions of even 1 digit, the sudoku will then not have a solution.
<br>
Thus, the sudoku solver does not work in practical use, but if used with an already trained model of much higher accuracy, one can get the solution to the sudoku.
