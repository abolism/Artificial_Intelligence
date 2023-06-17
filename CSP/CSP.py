

class CSP:
    def __init__(self, number_of_marks):
        """
        Here we initialize all the required variables for the CSP computation,
        according to the number of marks.
        """



        # Your code here
        self.number_of_marks = number_of_marks
        self.current_length = 0  # Update this line
        self.variables = []  # Update this line
        self.differences = [[]]  # Update this line

    def assign_value(self, i, v):
        """
        assign a value v to variable with index i
        """


        # Your code here

        self.variables[i] = v

        # pass

    def check_constraints(self) -> bool:
        """
        Here we loop over the differences array and update values.
        Meanwhile, we check for the validity of the constraints.
        """
        # Your code here
        toCheck = []
        for i in range(self.number_of_marks):

            for j in range(i+1,self.number_of_marks):
                self.differences[i][j] = self.variables[j] - self.variables[i]

                if self.differences[i][j] in toCheck:
                    return False
                toCheck.append(self.differences[i][j])
        return True


        # pass

    def backtrack(self, i):
        """
         In this function we should loop over all the available values for
         the variable with index i, and recursively check for other variables values.
        """
        # Your code here

        for j in range(1,self.current_length):
            self.assign_value(i,j)
            if self.check_constraints():
                if i == self.number_of_marks-1:
                    return True
                else:
                    if self.backtrack(i+1):
                        return True
        return False

        # pass

    def forward_check(self, i):
        """
        After assigning a value to variable i, we can make a forward check - if needed -
        to boost up the computing speed and prune our search tree.
        """
        # Your code here



# def check_length(self) -> bool:
#     # Your code here
#     for i in range(self.number_of_marks):
#         for j in range(i+1,self.number_of_marks):
#             if self.differences[i][j] > self.current_length:
#                 return False
#     return True





        # pass

    def find_minimum_length(self) -> int:
        """
        This is the main function of the class.
        First, we start by assigning an upper bound value to variable current_length.
        Then, using backtrack and forward_check functions, we decrease this value until we find
        the minimum required length.
        """

        """
        we wanna do the following in the golomb ruler problem.
        First, we start by assigning an upper bound value to variable current_length.
        Then, using backtrack and forward_check functions, we decrease this value until we find
        the minimum required length.
        """
        #we wanna subtract the max value from the min value in the differences array
        # Your code here
        self.current_length = self.number_of_marks * (self.number_of_marks - 1) // 2
        self.variables = [0] * self.number_of_marks
        self.differences = [[0] * self.number_of_marks for i in range(self.number_of_marks)]
        while not self.backtrack(1):
            self.current_length += 1
        return self.current_length
        # self.current_length = max(self.differences) - min




        # # Your code here
        # while True:
        #     pass
        # return 0

    def get_variables(self) -> list:
        """
        Get variables array.
        """
        # No need to change
        return self.variables
