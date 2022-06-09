import gpt2forbot

current_model = "myModel" #biggest = 1558M, smallest = 117M
gpt2 = gpt2forbot.GPT2(model_name=current_model)


print("\n\n-----------------------------------------Model Text Tokenizer test")

tokens = gpt2.enc.encode("yo yo yo yo blah blah blah")
print(tokens)
words_detokened = gpt2.enc.decode(tokens)

print(words_detokened) # should be the same as the text

# exit()  # exit here because only testing the tokenizer







print("\n\n-----------------------------------------Model Text Generation test")

initial_context ="winter is great. i wish it didn't get so cold sometimes though."
result = gpt2.generate_conditional(raw_text=initial_context, last_length=40)
print("Context=", initial_context)
print("Result=", result)




# for fine tuning run:

# python train.py --dataset dataset_filename.txt  --run_name mymodel

# (where dataset_filename.txt is a text file or raw text sentences.)

# after fine-tuning, a new model will be saved in ../models ... Change the line 3 to use the tuned model.