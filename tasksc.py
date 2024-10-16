
class tasks():
    def __init__(self):
        print(f"""
hier sind paar tasks 01 bis 07
you can make obyect from each Task
a = task01()
a.help()
a.run()
0
""")
        

    def task01(self):
        def help(self):
            print (f"""
    Task 1: Build a Financial Analyst System Using LangChain and Ollama
    Requirements:
    Use LangChain and Ollama to create a financial analyst system.
    The system should accept user queries related to finance (e.g., stock analysis, market trends, company reports).
    It should be able to analyze, process, and provide useful insights or answers from financial data.
1
    """)
            
        def run(self):
            print(f" this Task is still in work")
        return  help, run

    ##########################################################################
            

    def task02(self):
        def help(self):
            print(f"""
    Task 2: Image Classification Using Transfer Learning and a Built-in Dataset
    Requirements:
    Use any transfer learning model (e.g., VGG16, ResNet, or MobileNet) to build a neural network for image classification.
    Select a built-in dataset for image classification (e.g., CIFAR-10, MNIST, or Fashion MNIST).
    Preprocess the dataset and ensure it’s ready for input into the model.
    Fine-tune the pre-trained model by replacing the final layers to suit the classification task.
    Train and evaluate the model on the dataset, reporting accuracy and loss.

    """)
        def run(self):
            import keras
            from keras.datasets import fashion_mnist
            from keras.models import Sequential
            from keras.layers import Dense, Flatten
            from keras.utils import to_categorical

            keras.datasets.fashion_mnist.load_data()

            # 1. Load the .fashion_mnist Dataset
            (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

            # 2. Preprocess the Data
            # Normalize the data by scaling pixel values to be between 0 and 1
            X_train = X_train.astype('float32') / 255
            X_test = X_test.astype('float32') / 255

            # Initialize the Sequential model
            from tensorflow.keras import layers
            model_cnn = Sequential()  #models.

            # Add layers step-by-step using the add() method
            model_cnn.add(layers.InputLayer(input_shape=(28, 28, 1)))  # Input layer for 32x32x3 color images
            model_cnn.add(layers.Conv2D(32, (3, 3), activation='relu'))  # First Conv2D layer
            model_cnn.add(layers.MaxPooling2D((2, 2)))  # First MaxPooling layer

            model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Second Conv2D layer
            model_cnn.add(layers.MaxPooling2D((2, 2)))  # Second MaxPooling layer

            model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Third Conv2D layer

            model_cnn.add(layers.Flatten())  # Flatten the feature maps
            model_cnn.add(layers.Dense(64, activation='relu'))  # Fully connected layer
            model_cnn.add(layers.Dense(10, activation='softmax'))  # Output layer for 10 classes


            # Compile the model
            model_cnn.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',   #sparse_categorical_crossentropy  ..> only wenn no encoding for y_train
                          metrics=['accuracy'])

            history = model_cnn.fit(X_train, y_train, epochs=10, validation_split=0.2)

            # Evaluate the model
            test_loss, test_acc = model_cnn.evaluate(X_test, y_test)
            test_acc


            import matplotlib.pyplot as plt

            # Display the first test image
            plt.imshow(X_test[0]) #, cmap='color')
            plt.title("First Test Image")
            plt.show()

            # Plot training and validation accuracy/loss
            plt.figure(figsize=(12, 4))
            class_names = ['Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.show()
            # Display the first 5 test images and predicted labels
            predictions = model_cnn.predict(X_test[:10])


            # Plot training and validation accuracy/loss
            plt.figure(figsize=(12, 4))
            class_names = ['Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.show()
            # Display the first 5 test images and predicted labels
            predictions = model_cnn.predict(X_test[:10])

            plt.figure(figsize=(10, 2))
            for i in range(5):
                plt.subplot(1, 5, i+1)
                plt.imshow(X_test[i])
                plt.xticks([])
                plt.yticks([])
                #plt.title(f"Pred: {class_names[predictions[i].argmax()]}")
            plt.show()
        return  help, run
    #########################################################################################
            
    def task03(self):
        def help(self):
            print(f"""
    Task 3:Convert Sentences to Embeddings Using Ollama and Implement Similarity Search with a Vector Database
    Requirements:
    Use Ollama to convert three sample sentences into embeddings.
    Store the embeddings in a vector database (e.g., FAISS or Pinecone).
    Implement a similarity search function to compare the embeddings and return the most similar sentence for a given query.
    Test the system by inputting a new sentence and retrieving the most semantically similar one from the stored sentences.

    """)
            
        def run(self):
                        
            # generate embeddings 
            from sentence_transformers import SentenceTransformer

            # Load a pre-trained sentence transformer model
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # # Example words to embed
            # words = ["apple", "banana", "car", "train"]

            # Sample sentences to embed
            texts = [ "LangChain is a framework for building applications using LLMs.", 
            "Ollama provides easy access to various AI models.", 
            "Generative AI is revolutionizing industries." ]


            # Generate embeddings
            embeddings = model.encode(texts)


            # Print the embeddings
            for text, embedding in zip(texts, embeddings):
                print(f"Word: {text}, Embedding: {embedding}")

            from qdrant_client import QdrantClient
            client = QdrantClient(url="http://localhost:6333")
            from qdrant_client.models import Distance, VectorParams

            #Create a new collection for the words
            collection_name = "Pinecone"
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.DOT)  # Use the size of the embedding
            )

            from qdrant_client.http.models import PointStruct
            points = [
                PointStruct(
                    id=i + 1,  # Assign a unique ID to each point
                    vector=embedding.tolist()  # Convert embedding to a list for insertion
                )
                for i, embedding in enumerate(embeddings)  # Loop through embeddings and create PointStruct
            ]

            client.upsert(
                collection_name=collection_name,
                points=points  # Insert all points into the collection
            )
        return  help, run
    ##############################################################################################
            

    def task04(self):
        
        def help(self):
            print(f"""
    Task 4 : Build a Sentiment Analysis System Using LangChain and Ollama
    Requirements:
    Use LangChain integrated with Ollama to build a sentiment analysis system.
    The system should classify text inputs (e.g., product reviews, social media posts) into positive, negative, or neutral sentiment.
    Use a pre-trained model (ollama) to handle text embedding and sentiment classification.

    """)
            
        def run(self):
            from langchain_ollama.llms import OllamaLLM
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain

            # 01 Initialize Ollama model
            ollama = OllamaLLM(model="llama3.2") 

            # 02 Define a LangChain prompt template for sentiment analysis
            template = """
            Classify input text into: positive, negative, or neutral Text: '{text}'"
            """
            prompt = PromptTemplate(input_variables=["text"], template=template)

            # 03 Creating an LLM chain with Ollama and the prompt template
            chain = LLMChain(llm=ollama, prompt=prompt)

            # 04 using model to handle text embedding and sentiment classification
            input_text = "good product! It works very good"  #"bad product! It dosnt work "
            result = chain.run(input_text)
            print(f"Sentiment: {result}")
        return  help, run

    ########################################################################
            
    def task05(self):
        def help(self):
            print(f"""

    """)
            
        def run(self):
                    
            from langchain_ollama.llms import OllamaLLM

            llm = OllamaLLM(model="llama3.2",model_kwargs={"temperature": 1, "max_tokens": 200})
            while True:
                prompt = input("Ask your question, or type 'exit' ")
                if prompt.lower() == "exit":
                    print("Exit")
                    break
                answer = llm(prompt)
                print(answer)

            from langchain_ollama.llms import OllamaLLM

            # 01 Initialize the Ollama LLM
            llm = OllamaLLM(model="llama3.2")  #model="llama3.2", model="deepseek-coder-v2", model="qwen2"

            # Make a request to the model
            response = llm("please find the 10 germany movies in historz, return data as json")
            print(response)
            

            

            # Task 5 part 2
            from llama_index.llms.ollama import Ollama
            from llama_index.core.llms import ChatMessage

            llm = Ollama(model="llama3.2",model_kwargs={"temperature": 1, "max_tokens": 200})
            messages = [ ChatMessage(role="user", content= prompt)]
            while True:
                prompt = input("Ask your question, or type 'exit' ")
                if prompt.lower() == "exit":
                    print("Exit")
                    break
                answer = llm.chat(messages)
                print(answer)

            from llama_index.llms.ollama import Ollama
            from llama_index.core.llms import ChatMessage

            # Initialize the LLM
            llm = Ollama(model="llama3.2", model_kwargs={"temperature": 1, "max_tokens": 200})

            while True:
                prompt = input("Ask your question, or type 'exit': ")
                if prompt.lower() == "exit":
                    print("Exit")
                    break
                
                # Add the user message to the messages list
                messages = [(ChatMessage(role="user", content=prompt))]
                
                # Get the answer from the LLM
                answer = llm.chat(messages)
                print(answer)
            

            ### Chat Streaming 
            from llama_index.llms.ollama import Ollama
            from llama_index.core.llms import ChatMessage

            # Initialize the model
            llm = Ollama(model="llama3.2")

            messages = [
                ChatMessage(
                    role="system", content="You are helpful assistant to create programs"
                ),
                ChatMessage(role="user", content="Write a python program to calculate the fact of numbers"),
            ]

            response = llm.stream_chat(messages)
            for r in response:
                print(r.delta, end="")
                

            from llama_index.llms.ollama import Ollama
            from llama_index.core.llms import ChatMessage

            # Initialize the model
            llm = Ollama(model="llama3.2")

            messages = [
             #   ChatMessage(
              #      role="system", content="You are helpful assistant to create programs"
             #   ),
                ChatMessage(role="user", content="please find the 10 germany movies from 2000, return data as json"),
            ]

            response = llm.chat(messages)
            print(response)
        return  help, run
            
    ################################################################################
     
    def task06(self):
        def help(self):
            print(f"""
    Task 6 : Big Project
    Requirements:
    integrate the last 5 projects in one main class , each project in a function inside this class and add a main function which starts asking the user which app he wants to run if the user press the app numbers (like the game we did before)

    """)
        def run(self):
            print(f" this Task is don")
        return  help, run


    ################################################################################
            
    def task07(self):
        def help(self):
            print(f"""
    Task 7 Text Summarization using LangChain and Ollama
    Requirements: 
    The goal is to use LangChain and the Ollama to build a python script that takes a block of text and summarizes it into a concise version. 
    Build the system again using llamaindex 

    """)
            
        def run(self):
                    
            #!pip install upgrade langchain

            from langchain.llms import Ollama
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            from langchain_ollama.llms import OllamaLLM

            # 01 Initialize the Ollama LLM
            llm = OllamaLLM(model="llama3.2")  #model="llama3.2", model="deepseek-coder-v2", model="qwen2"

            # Define the prompt template for summarization
            prompt_template = PromptTemplate(
                input_variables=["text"],
                template="Please summarize the following text:\n\n{text}\n\nSummary:"
            )

            # Create an LLMChain for summarization
            llm_chain = LLMChain(llm=model, prompt=prompt_template)

            text_to_summarize = """ Generative AI refers to a category of artificial intelligence that focuses on creating new content based on learned patterns 
            from existing data. Unlike traditional AI, which typically analyzes data to perform specific tasks, generative AI goes a step further by generating 
            original outputs, such as text, images, music, and even video. Machine Learning Models: Generative AI relies heavily on machine learning techniques, 
            particularly deep learning. Models like Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) are commonly used to produce new 
            data by understanding the underlying distributions of the input data. Applications: The applications of generative AI are vast and growing. 
            In creative fields, it can be used to generate art, music, and literature. In the business sector, it helps in creating realistic simulations 
            for training purposes or generating marketing content. Additionally, it plays a significant role in game development by producing unique characters 
            and environments. Text Generation: One of the most prominent areas of generative AI is text generation. Models like GPT (Generative Pre-trained 
            Transformer) are trained on large corpuses of text and can produce human-like responses, write essays, summarize information, and even code. 
            These models have been used in chatbots, virtual assistants, and content creation tools. Ethical Considerations: The rise of generative AI 
            also raises ethical questions. Concerns about copyright, misinformation, and the potential for misuse (such as generating deepfakes) have led 
            to discussions about the responsible use of this technology. Ensuring transparency and accountability in generative AI systems is critical to 
            addressing these issues. Future Directions: As generative AI continues to evolve, we can expect advancements that improve the quality and diversity
            of generated content. The integration of generative AI with other technologies, such as augmented reality (AR) and virtual reality (VR), may lead
            to immersive experiences that blur the lines between reality and creation. In summary, generative AI represents a significant advancement
            in the field of artificial intelligence, enabling machines to create original content and push the boundaries of creativity and innovation. 
            Its impact is felt across multiple industries, prompting ongoing discussions about its ethical use and future potential.
            """
            # Get the summary
            summary = llm_chain.run(text_to_summarize)
            print("Original Text:\n", text_to_summarize)
            print("\nSummary:\n", summary)

            #!pip install llama-index upgrade
        return  help, run
