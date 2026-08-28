from local_llm import generate_answer


question = "What information is available in the document?"


text_results = [
    "A large tree stands in an open field while warm sunlight shines over the landscape."
]


image_results = [
    "A series of photos showing different landscapes."
]


answer = generate_answer(
    question,
    text_results,
    image_results
)


print("\n========== FINAL ANSWER ==========")
print(answer)