# Chat with PDF Documents

This is a Streamlit-based web application that allows users to upload multiple PDF documents, extract text, and then query this information through a conversational interface powered by Google Generative AI.
The Users can ask any questions related to the documents uploaded.

![Demo](demo.gif)

To build this application we follow certain steps which are:
1. The first step is to extract text content from all uploaded PDF files and concatenate it into a single variable. This involves using the `PyPDF2` library to read each PDF file, extract text from its pages, and aggregate this text into a comprehensive string.
2. Next, the extracted text is divided into smaller chunks using the `RecursiveCharacterTextSplitter` from the `langchain` library. This step breaks down the large text into manageable segments for further processing and analysis.
3. After obtaining text chunks, the next step likely involves generating semantic vectors (embeddings) for each chunk using `GoogleGenerativeAIEmbeddings` from the `google-generativeai` library. These embeddings capture the semantic meaning and relationships within the text data.
4. Finally, integrate the generated vector store with a conversational interface for question answering. This involves using `langchain` and `ChatGoogleGenerativeAI` to build a conversational chain that processes user queries based on the semantic vectors derived from the PDF text chunks.
5. Use `Streamlit` to create an interactive web interface for uploading PDF documents and querying information.


## Features

- **Upload PDF Documents**: Users can upload one or more PDF files where the size of each file should be less than 200mb.
- **Text Extraction**: Automatically extract text from uploaded PDF documents.
- **Text Chunking**: Break down extracted text into manageable chunks for efficient processing.
- **Semantic Vector Store**: Utilize Google Generative AI embeddings to generate semantic vectors for text chunks.
- **Conversational Interface**: Engage in a chat-like conversation with the application to query information from the uploaded documents.


Firstly, we need to install all the dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Running the Application

For start the application we run the following command in the terminal:

```bash
streamlit run app.py
```
This will launch the application locally. You can access it in your web browser at [http://localhost:8501](http://localhost:8501).

### 2. Upload PDF Documents

Use the file uploader on the web interface to upload one or more PDF documents of size less than 200mb.

### 3. Extract Text and Generate Semantic Vector Store

After uploading PDFs, click on the "Collect Information" button. This will extract text from the uploaded documents and generate a semantic vector store using Google Generative AI embeddings.

### 4. Query Information

Once the text extraction is complete, you can enter your questions or queries in the text input box provided. Press Enter to receive a response based on the uploaded documents.




## Configuration

Before running the application, we must ensure we have set up the required environment variables:

- `GOOGLE_API_KEY`: API key for Google Generative AI. This should be stored in a `.env` file in the root directory of the project.

## Dependencies

- `streamlit`: For building the interactive web interface.
- `PyPDF2`: For extracting text from PDF files. It allows to extract text, merge and split PDF documents
- `langchain`: For text chunking, semantic vectorization, question-answering and conversational AI capabilities. (basically for NLP tasks)
- `dotenv`: For loading environment variables from a `.env` file.
- `faiss-cpu`: For efficient similarity search and clustering of dense vectors. The faiss-cpu variant is optimized for CPU-based computations.
- `google-generativeai`: It provides access to Google's Generative AI models and tools for natural language understanding and generation tasks.
- `langchain_google_genai`: A component within the langchain framework that integrates with Google's Generative AI tools and models.

## Contributing

Contributions are welcome! Please feel free to fork this repository and submit pull requests to propose changes or improvements.


## Alternatives

While this project uses specific libraries and tools, there are alternative options we might consider for similar tasks:

- **PDFMiner**: An alternative library for extracting text from PDF files.
- **spaCy**: Another popular NLP library that offers text processing capabilities.
- **Hugging Face Transformers**: Provides access to various pre-trained language models for tasks like question answering and text generation.

