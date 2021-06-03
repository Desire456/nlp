import text_preprocessing

input_str = input("Input something: ")

input_str = text_preprocessing.preprocess(input_str)

print(input_str)
print(text_preprocessing.pos_tag('And now for something completely different'))
