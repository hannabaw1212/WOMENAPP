# import torch
# from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, Trainer, TrainingArguments
# from sklearn.model_selection import train_test_split
# import pandas as pd
# torch.manual_seed(42)


# data = pd.read_csv('data/QA_data.csv')

# tokenizer = DistilBertTokenizerFast.from_pretrained('./fine_tuned_model')
# model_qa = DistilBertForQuestionAnswering.from_pretrained('./fine_tuned_model')
# contexts = list(pd.read_csv('data/predefined_contexts.csv').iloc[:,1])

# train_df, eval_df  = train_test_split(data, test_size = 0.1, random_state=42)

# train_dataset = train_df.to_dict(orient="records")
# eval_dataset = eval_df.to_dict(orient="records")
# def preprocess_data(data):
#     encoding = tokenizer(
#         data['Question'],
#         data['Paragraph_Context'],
#         truncation = True,
#         padding = "max_length",
#         max_length = 512
#     )
#     return {
#           "input_ids": encoding["input_ids"],
#         "attention_mask": encoding["attention_mask"],
#         "start_positions": data["Answer_Start_Position"],
#         # "end_positions": data["Answer_End_Position"],
#         "end_positions": data["Answer_Start_Position"] + len(data['Paragraph_Context']) - 1
#     }

# def answer_question(question):
#     best_answer = None
#     best_score = float("-inf")

#     for context in contexts:
#         inputs = tokenizer(question, context, return_tensors="pt", padding=True, truncation=True)

#         with torch.no_grad():
#             outputs = model_qa(**inputs)
#             start_scores, end_scores = outputs.start_logits, outputs.end_logits

#         start_pos = torch.argmax(start_scores)
#         end_pos = torch.argmax(end_scores)
#         score = start_scores[0, start_pos] + end_scores[0, end_pos]

#         if score > best_score:
#             best_score = score
#             answer_tokens = inputs["input_ids"][0][start_pos:end_pos + 1]
#             best_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer_tokens))

#     return best_answer

# processed_train_dataset = [preprocess_data(data) for data in train_dataset]
# processed_eval_dataset = [preprocess_data(data) for data in eval_dataset]


# # train_args = TrainingArguments(
# #     output_dir = "./results",
# #     per_device_eval_batch_size = 1, 
# #     num_train_epochs = 15,
# #     logging_steps=1,
# #     evaluation_strategy="epoch",
# #     logging_dir = "./logs",
# #     per_device_train_batch_size = 4
# # )

# train_args = TrainingArguments(
#     output_dir="./results",
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=1,
#     num_train_epochs=11,
#     logging_steps=10,
#     evaluation_strategy="epoch",
#     logging_dir="./logs",
#     learning_rate=5e-5,
#     seed=42,
# )

# trainer = Trainer(
#     model = model_qa,
#     args = train_args,
#     train_dataset = processed_train_dataset,
#     eval_dataset = processed_train_dataset
# )

# # trainer.train()

# # # Save the fine-tuned model
# # model_qa.save_pretrained("./fine_tuned_model")
# # tokenizer.save_pretrained("./fine_tuned_model")

# print("Done training")

# #############TEST#################







# # # Predefined list of contexts
# # contexts = [
# #     "Ehlers-Danlos Syndrome (EDS) is a group of inherited disorders that affect the connective tissues in the body. These tissues, which include skin, tendons, ligaments, and blood vessel walls, are responsible for providing strength and elasticity to the body's structures. There are 13 subtypes of EDS, each with its own specific set of symptoms and characteristics.",
# #     "Symptoms of Ehlers-Danlos Syndrome can vary widely depending on the subtype, but some common symptoms include hypermobility of joints, easily bruised or damaged skin, chronic pain, and fatigue. Some individuals with EDS may also experience gastrointestinal issues and heart valve problems.",
# #     "There is no cure for Ehlers-Danlos Syndrome, but treatment primarily focuses on managing symptoms and preventing complications. This may include physical therapy to strengthen muscles and stabilize joints, pain management, and lifestyle modifications to reduce the risk of injury. In some cases, surgery may be necessary to repair joint damage or treat other complications.",
# # ]

# # # Get the question from the user
# # question = "What is Pain managment in EDS?"

# # # Use the answer_question function to get the answer
# # # index 53
# # answer = answer_question(question)
# # print("Answer2:", answer)



# # output_dir: Keep the same ("./results"). This is a user-specific preference and does not impact model performance.

# # per_device_train_batch_size: Set to 8. A larger batch size can help with generalization and speed up training. However, if you encounter memory issues, you can reduce the batch size (e.g., to 4).

# # per_device_eval_batch_size: Set to 4. A larger evaluation batch size can speed up the evaluation process without significantly impacting memory usage.

# # num_train_epochs: Set to 5. With a smaller dataset, the model might learn more quickly, so fewer epochs might be sufficient. Start with 5 epochs and monitor the model's performance. If it continues to improve, you can increase the number of epochs.

# # logging_steps: Set to 10. This is a user-specific preference and doesn't impact model performance. Adjust based on your desired frequency of training updates.

# # evaluation_strategy: Keep as "epoch". Evaluating after each epoch provides a good balance between monitoring progress and computational cost.

# # logging_dir: Keep the same ("./logs"). This is a user-specific preference and does not impact model performance.

# # learning_rate: Set to 5e-5. A larger learning rate can speed up training but may result in unstable training. It's a trade-off, so you might need to experiment with different learning rates (e.g., 3e-5, 5e-5, or 1e-4) to find the best value for your dataset.

# # seed: Keep the same (42). This is an arbitrary value for reproducibility, so you can choose any fixed value.

