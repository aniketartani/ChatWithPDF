import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
# from streamlit_extras.add_vertical_space import add_vertical_space
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from streamlit_chat import message
import pandas as pd
import firebase_admin
from firebase_admin import credentials, storage
from gtts import gTTS
from pageadd import add_page_numgers
import speech_recognition as sr
# from googletrans import Translator
# import httpcore
# setattr(httpcore, 'SyncHTTPTransport', 'AsyncHTTPProxy')

# Initialize the recognizer
recognizer = sr.Recognizer()
# translator = Translator()

# Initialize Firebase with credentials
# cred = credentials.Certificate("pdf-test-e24d4-firebase-adminsdk-bbz2s-29a5baf43b.json")
# firebase_admin.initialize_app(cred, {
#     'storageBucket': 'pdf-test-e24d4.appspot.com'
# })
from io import BytesIO
import firebase_admin
from firebase_admin import credentials
import time

def download_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()
    st.write("Downloaded file: ", os.path.basename(file_path))
    st.download_button(label="Download PDF", data=pdf_bytes, file_name=os.path.basename(file_path), mime="application/pdf")


def delete_pdf(pdf_name):
    # Get a reference to the Firebase Storage bucket
    bucket = storage.bucket('pdf-test-e24d4.appspot.com')

    # Construct the path to the PDF in Firebase Storage
    pdf_path = f'{pdf_name}'  # Modify this path according to your Firebase Storage structure

    # Check if the PDF exists in Firebase Storage
    blob = bucket.blob(pdf_path)
    os.remove(pdf_path)
    if blob.exists():
        # Delete the PDF from Firebase Storage
        blob.delete()
        print(f"PDF '{pdf_name}' deleted successfully!")
    else:
        print(f"PDF '{pdf_name}' does not exist!")
def upload_pdf(file_path, destination_blob_name):
    """Uploads a PDF file to Firebase Storage."""
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    print(f'File {file_path} uploaded to {destination_blob_name}.')
# Initialize the Firebase app
# cred = credentials.Certificate("pdf-test-e24d4-firebase-adminsdk-bbz2s-29a5baf43b.json")
# firebase_admin.initialize_app(cred, name='aniket', options={
#     'storageBucket': 'pdf-test-e24d4.appspot.com'
# })

if not firebase_admin._apps:
    cred = credentials.Certificate('pdf-test-e24d4-firebase-adminsdk-bbz2s-29a5baf43b.json') 
    default_app = firebase_admin.initialize_app(cred)

# Now you can proceed with your storage operations


bucket = storage.bucket('pdf-test-e24d4.appspot.com')
# time.sleep(5)
with st.sidebar:
    st.title("Convert Your Pdf to Num")
    pdf2 = st.sidebar.file_uploader("Upload", type='pdf')
    if pdf2 is not None:
        file_path = 'file.pdf'
        num="num"
        destination_blob_name = f'pdfs/{pdf2.name}.pdf'
        pdf=pdf2
        # upload_pdf(file_path, destination_blob_name)
    # Save the uploaded PDF file to the 'pdfs' folder
        with open(os.path.join(pdf.name), 'wb') as f:
            f.write(pdf.getbuffer())
            # add_page_numgers(pdf.name)
            # add_page_numgers(pdf.name)
        add_page_numgers(pdf.name)    
        print("PDF NAME IS ",pdf.name)
        # if st.button("Download File"):
        file_path=f'{pdf.name}.pdf'
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        st.success("Converted file")
        st.download_button(label="Download PDF", data=pdf_bytes, file_name=os.path.basename(file_path), mime="application/pdf")
        #count
        
        # print("Total count is",total_count)
        
        # upload_pdf(os.path.join(pdf.name+".pdf"), destination_blob_name)   
    st.title("Chat with PDF 💬")
    loaded_chat_string="testing"
    # st.button("test")
    load_dotenv()

def main():
    # upload a PDF file
    
    if 'selected_file_path' not in st.session_state:
        print("sessions",st.session_state)
        st.session_state.selected_file_path = ""  # Set default index to 0 or any appropriate default value

# Create the radio button widget
    min = st.number_input("Minimum", value=None, placeholder="Type a min num")
    # st.write("The current number is ", min)
    max = st.number_input("Maxmium", value=None, placeholder="Type a max num")
    st.write(f"Chat between pages {min} and {max}")
    pdf = st.sidebar.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:
        file_path = 'file.pdf'
        num="num"
        destination_blob_name = f'pdfs/{pdf.name}.pdf'
        # upload_pdf(file_path, destination_blob_name)
    # Save the uploaded PDF file to the 'pdfs' folder
        with open(os.path.join(pdf.name), 'wb') as f:
            f.write(pdf.getbuffer())
            # add_page_numgers(pdf.name)
            # add_page_numgers(pdf.name)
        # add_page_numgers(pdf.name)    
        print("PDF NAME IS ",pdf.name)
        
        upload_pdf(os.path.join(pdf.name), destination_blob_name)    
    # if st.sidebar.button("test"):
 

        # with open(os.path.join('pdfs', pdf.name), 'rb') as f:
        #     pdf_bytes = f.read()
        # upload_pdf_to_firebase(pdf_bytes, pdf.name)    
    #     with open('pdfs/' + pdf.name, 'rb') as f:
    # # Upload the file to Firebase Storage
    #         blob = bucket.blob(pdf.name)
    #         blob.upload_from_file(f)
    # pdf = st.sidebar.file_uploader("Without numbered PDF", type='pdf')
    # if pdf is not None:
    #     file_path = 'file.pdf'
    #     destination_blob_name = f'pdfs/{pdf.name}'
    #     # upload_pdf(file_path, destination_blob_name)
    #     # Save the uploaded PDF file to the 'pdfs' folder
    #     with open(os.path.join(pdf.name), 'wb') as f:
    #         f.write(pdf.getbuffer())
    #     upload_pdf(os.path.join(pdf.name), destination_blob_name)  
    #     print("File uploaded to Firebase Storage successfully.")

    # pdf_files = os.listdir('pdfs')
    # pdf_files = [file for file in os.listdir('pdfs') if file.endswith('.pdf')]
    blobs =bucket.list_blobs()

# Filter PDF files
    pdf_files = [blob.name for blob in blobs if blob.name.endswith('.pdf')]
    # selected_file_index = st.sidebar.radio("Select a PDF file", range(len(pdf_files)), format_func=lambda i: pdf_files[i] if len(pdf_files) > i else "")
    selected_file_index = st.sidebar.radio("Select a PDF file", range(len(pdf_files)), format_func=lambda i: pdf_files[i] if len(pdf_files) > i else "") 
    print(selected_file_index)
    # selected_file_index = st.sidebar.button("Select a PDF file", pdf_files, format_func=lambda i: pdf_files[i] if len(pdf_files) > i else None)
    # selected_file_index = 0
    # # Inject CSS style into Streamlit
    # st.markdown("""
    # <style>
    # div.stButton button {
    #     background-color: white;
    #     width: 250px;
    # }
    # </style>
    # """, unsafe_allow_html=True)
    # for i in range(len(pdf_files)):
    #     # button_label = f"<div style='width: 200px; height: 50px; display: flex; align-items: center; justify-content: center;'>{pdf_files[i]}</div>"
    #     if st.sidebar.button(os.path.basename(pdf_files[i])):
    #         selected_file_index = i
            # print("SIDE BUTTON",selected_file_index)
    if len(pdf_files) > selected_file_index:
        # selected_file_path = os.path.join('pdfs', pdf_files[selected_file_index])
        selected_file_path=pdf_files[selected_file_index]
        if st.session_state.selected_file_path != selected_file_path:
            # print("changed pdff",st.session_state.selected_file_path)
            # print(type(st.session_state.selected_file_path))
            # print(selected_file_path)
            if "messages" in st.session_state:
                st.session_state.messages = []
                if os.path.exists(os.path.join(f"{pdf}.pkl")):
                    os.remove(selected_file_path)      
                    os.remove(f"{store_name}.pkl")  

        # st.write("Selected file path:", selected_file_path)
        # print(pdf)
        # if uniquecode==selected_file_path:
        #     print("same string")
        pdf=selected_file_path
        print("name is",pdf)
        st.session_state.selected_file_path=pdf
        print("second",st.session_state.selected_file_path)
        pdf_reader=""
        print(pdf_reader)
        #FIREBASE download
        blob = bucket.blob(pdf)
        blob2=bucket.blob(f'{pdf}.pkl')
        print(blob2)
        temp_file = "file.pdf"  # Local path to temporarily store the downloaded PDF
        blob.download_to_filename(pdf)    
        # blob.download_to_filename(f'{pdf}.pkl')    
        print("File downloaded successfully.")        
#         with open('file.pdf', 'rb') as file:
#             pdf_reader = PdfReader(file)
#         # with open(temp_file, "rb") as file:
#         #     from PyPDF2 import PdfReader
#         #     # pdf_reader = PdfReader(file)
#         #     pdf_reader = PdfReader(file)
#     # Use any PDF processing library like PyPDF2 or pdfplumber
#     # Example using PyPDF2 to extract text from the PDF

#         # Read the selected PDF file
#         pdf_reader = PdfReader(selected_file_path)
# #         # os.remove("file.pdf")
#         print(pdf_reader)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )
#         print("chunking",pdf)
#         # uniquecode=pdf
        if os.path.exists(os.path.join(f"{pdf}.pkl")):
            print("already present")
            pdf_reader = PdfReader(selected_file_path)
    #         # os.remove("file.pdf")
            print(pdf_reader)
            count=0
            text = ""
            for page in pdf_reader.pages:
                count=count+1
                print(count)
                if count>min and count<max:
                    print("doneee")
                    text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            print("chunking",pdf)            
            chunks = text_splitter.split_text(text=text)
            print("doing chunkimg")            
            # chunks = text_splitter.split_text(text=text)
        else:    
            pdf_reader = PdfReader(selected_file_path)
    #         # os.remove("file.pdf")
            print(pdf_reader)
            count=0
            text = ""
            for page in pdf_reader.pages:
                count=count+1
                print(count)
                if count>min and count<max:
                    print("doneee")
                    text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            print("chunking",pdf)            
            chunks = text_splitter.split_text(text=text)
            print("doing chunkimg")

#         #REMOVE PDF
        if st.sidebar.button("X"):
            delete_pdf(pdf)
            st.experimental_rerun() 

#         # Store name
        if pdf is not None:
            store_name = pdf
            print("store name is ", store_name)
            # Load or compute embeddings
            if os.path.exists(os.path.join(f"{store_name}.pkl")):
                print("chossing from already present")
                with open(os.path.join(f"{store_name}.pkl"), "rb") as f:
                    VectorStore = pickle.load(f)
                print("making new")
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)
                    upload_pdf(f"{store_name}.pkl", f'{store_name}.pkl')    
                    
            else:
                # Compute embeddings
                print("making new")
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)
                    upload_pdf(f"{store_name}.pkl", f'{store_name}.pkl')
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # React to user input
            if prompt := st.chat_input("What is up?"):
                # Display user message in chat message container
                st.chat_message("user").markdown(prompt)
                docs = VectorStore.similarity_search(query=prompt, k=3)
                # print("DOCS IS ",docs)
                llm = langchain.llms.OpenAI(model="gpt-3.5-turbo-instruct")
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                response=""
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=prompt)
                    print(cb)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                # response = f"Echo: {prompt}"
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response + "\n\n---\n\nSource=" + str(docs))
                # Add assistant response to chat history
                    tts = gTTS(text=response, lang='en')
                    # hindi = translator.translate(response, src='en', dest='hi').text
                    # tts2=gTTS(text=hindi, lang='hi')
                    tts.save("response.mp3")
                    # tts2.save("responseh.mp3")
                    
                st.session_state.messages.append({"role": "assistant", "content": response})  
            if st.sidebar.button("Read in English"):
                    audio_file = open('response.mp3', 'rb')
                    audio_bytes = audio_file.read()
                    # translated_text = translator.translate(english_text, src='en', dest='hi').text
                    # Display the audio player
                    st.audio(audio_bytes, format='audio/mp3')
            # if st.sidebar.button("Read in Hindi"):
            #         audio_file = open('responseh.mp3', 'rb')
            #         audio_bytes = audio_file.read()
            #         # translated_text = translator.translate(english_text, src='en', dest='hi').text
            #         # Display the audio player
            #         st.audio(audio_bytes, format='audio/mp3')                    
            # if st.sidebar.button("Hindi"):
                   

# Initialize an empty list to store chat history



if __name__ == '__main__':
    # uniquecode=""
    main()