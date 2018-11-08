def print_result(correct=False):
    print('\n'*5 + '='*50 + '\n'*3)
    if correct:
        print("Congratulations! Your answer is correct.")
    else:
        print("Your answer appears to be incorrect. Try again!")
    print('\n'*3 + '='*50)