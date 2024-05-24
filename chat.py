import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('bookstore_intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bookie"

# Sample data for book recommendations and prices
book_recommendations = {
    "history": ["History of the World", "The Second World War", "Sapiens", "Guns, Germs, and Steel", "A People's History of the United States"],
    "fiction": ["To Kill a Mockingbird", "1984", "The Great Gatsby", "The Catcher in the Rye", "The Hobbit"],
    "mystery": ["The Girl with the Dragon Tattoo", "Gone Girl", "Big Little Lies", "In the Woods", "The Da Vinci Code"],
    "romance": ["Pride and Prejudice", "Jane Eyre", "Outlander", "The Notebook", "Me Before You"],
    "science fiction": ["Dune", "Ender's Game", "Neuromancer", "The Left Hand of Darkness", "Snow Crash"],
    "fantasy": ["Harry Potter and the Sorcerer's Stone", "The Lord of the Rings", "A Game of Thrones", "The Name of the Wind", "The Way of Kings"],
    "biography": ["The Diary of a Young Girl", "Steve Jobs", "Long Walk to Freedom", "The Wright Brothers", "Alexander Hamilton"],
    "self-help": ["How to Win Friends and Influence People", "Atomic Habits", "The Power of Habit", "Thinking, Fast and Slow", "The 7 Habits of Highly Effective People"],
    "thriller": ["The Silent Patient", "The Woman in the Window", "The Girl on the Train", "The Couple Next Door", "The Reversal"],
    "non-fiction": ["Educated", "Becoming", "Unbroken", "The Immortal Life of Henrietta Lacks", "Into the Wild"]
}

book_prices = {
    "1984": "$9.99",
    "Animal Farm": "$7.99",
    "Homage to Catalonia": "$12.99",
    "Down and Out in Paris and London": "$10.99",
    "The Road to Wigan Pier": "$11.99",
    # Add prices for other books here
}

bestsellers = ["To Kill a Mockingbird", "1984", "The Great Gatsby", "The Catcher in the Rye", "The Hobbit"]
pre_order_instructions = "Visit our pre-order page on the website and select the books you wish to pre-order."
new_arrivals = ["The Midnight Library", "Klara and the Sun", "The Four Winds", "Project Hail Mary", "The Last Thing He Told Me"]

shipping_options_responses = [
    "We offer standard, express, and overnight shipping. Costs vary by option.",
    "Standard shipping takes 5-7 business days. Expedited options are available at an extra cost.",
    "Yes, we ship internationally. Shipping costs vary based on location.",
    "Free shipping is available on orders over a certain amount. Check our shipping policy for details."
]

def get_books_by_genre(genre):
    return book_recommendations.get(genre.lower(), [])

def get_author_books(author):
    author_books_data = {
        "george orwell": ["1984", "Animal Farm", "Homage to Catalonia", "Down and Out in Paris and London", "The Road to Wigan Pier"]
        # Add more well-known authors and their books here
    }
    return author_books_data.get(author.lower(), [f"{author}'s Book 1", f"{author}'s Book 2", f"{author}'s Book 3"])

def get_book_price(book):
    return book_prices.get(book, "Price not available")

def generate_response(intent, sentence_tokens, session_memory):
    response = random.choice(intent['responses'])

    if "{genre}" in response:
        genre = None
        for word in sentence_tokens:
            if word.lower() in book_recommendations:
                genre = word.lower()
                break
        if genre:
            response = response.replace("{genre}", genre)
            books = get_books_by_genre(genre)
            if books:
                response = response.replace("{genre_books}", ", ".join(books[:5]))

    if "{book_list}" in response:
        response = response.replace("{book_list}", ", ".join(bestsellers))

    if "{bestsellers_list}" in response:
        response = response.replace("{bestsellers_list}", ", ".join(bestsellers))

    if "{pre_order_instructions}" in response:
        response = response.replace("{pre_order_instructions}", pre_order_instructions)

    if "{author}" in response:
        author = None
        for i in range(len(sentence_tokens)):
            if sentence_tokens[i].lower() == "by" and i + 1 < len(sentence_tokens):
                author = ' '.join([sentence_tokens[j].capitalize() for j in range(i + 1, len(sentence_tokens))])
                break
        if author:
            response = response.replace("{author}", author)
            author_books = get_author_books(author)
            if author_books:
                response = response.replace("{author_books}", ", ".join(author_books[:5]))
        else:
            response = response.replace("{author}", "the specified author")

    if "{book}" in response or "{price}" in response:
        book = None
        for word in sentence_tokens:
            capitalized_word = word.capitalize()
            if capitalized_word in book_prices:
                book = capitalized_word
                break
        if book:
            response = response.replace("{book}", book)
            price = get_book_price(book)
            response = response.replace("{price}", price)
        else:
            response = response.replace("{book}", "the book")
            response = response.replace("{price}", "Price not available")

    if "{new_books_list}" in response:
        response = response.replace("{new_books_list}", ", ".join(new_arrivals))

    if intent['tag'] == "shipping":
        last_response = session_memory.get('last_shipping_response', "")
        available_responses = [r for r in shipping_options_responses if r != last_response]
        if not available_responses:
            available_responses = shipping_options_responses
        response = random.choice(available_responses)
        session_memory['last_shipping_response'] = response

    return response

def respond_to_query(sentence, session_memory):
    sentence_tokens = tokenize(sentence)
    X = bag_of_words(sentence_tokens, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = generate_response(intent, sentence_tokens, session_memory)
                return response
    else:
        return "I do not understand. Could you please rephrase your question?"

print("How can I help you? (type 'quit' to exit)")
session_memory = {}
while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break

    response = respond_to_query(sentence, session_memory)
    print(f"{bot_name}: {response}")
