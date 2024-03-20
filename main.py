
# Implemented on March 10, 2024 
# run the serrver and use http://localhost:9090/docs to  test endpoints
# Importing required libraries
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile, status
import random, string
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
# For evaluation of RAG
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import uvicorn

load_dotenv()

app = FastAPI()

openai_api_key = os.getenv("OPENAI_API_KEY")
assert openai_api_key is not None, "OpenAI API key is not set"

pinecone_api_key = os.getenv("PINECONE_API_KEY")
assert pinecone_api_key is not None, "Pinecone API key is not set"

embed_model_name = os.getenv("EMBED_MODEL")


origins = ["*"]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
    )

def init_pinecone_index(index_name:str):
    pine_client = Pinecone(api_key=pinecone_api_key)
    serverless_spec = ServerlessSpec(cloud="aws", region = "us-west-2")
    for ind in pine_client.list_indexes():
        if ind['name'] == index_name:
            pine_client.delete_index(index_name)
            print("deleted previous index")
    try:         
        pine_client.create_index(name=index_name,
                                      dimension=384,
                                      spec=serverless_spec,
                                      metric='cosine')
        #rayo_index = pine_client.Index(index_name)
        #return rayo_index
    except Exception as e:
        raise Exception(f"Unable to create index. Error: {str(e)}")

def delete_previous_index(index_name:str):
    try:    
        pine_client = Pinecone(api_key=pinecone_api_key)
        for ind in pine_client.list_indexes():
            if ind['name'] == index_name:
                pine_client.delete_index(index_name)
                print("deleted previous index")
            #else:
            #    print(f"index name {index_name} not found. Safe to continue")    
    except Exception as e:
        raise Exception(f"Unable to delete index. Error: {str(e)}")            


embed_fn = HuggingFaceEmbeddings(model_name = embed_model_name)
chat_llm = ChatOpenAI(model = "gpt-3.5-turbo", openai_api_key = openai_api_key, temperature=0.1)
vecdb = None
qa_chain = None

@app.get("/", response_class=HTMLResponse)
def index():
    message = "Welcome to Rayo Assistive AI"
    html_content = f"<html><body><h1>{message}</h1></body></html>"
    return HTMLResponse(content=html_content)

@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    global qa_chain,vecdb
    contents = await file.read()
    #file_extension =file.filename[-4:].lower().replace(".", "")
    curr_date = str(datetime.now())
    #file_path = f"temp/{filename}"
    if contents:
        # Read  the PDF file  and split it into documents
        try:
            filename = "".join(random.choices(string.ascii_letters, k=4)) + curr_date.split('.')[0].replace(':', '-').replace(" ", 'T')
            with open(filename, 'wb') as tmp_file:
                tmp_file.write(contents)
            pdf_loader = PyPDFLoader(file_path=str(Path(__file__).parent.joinpath( filename)))
            docs = pdf_loader.load()    
            #return JSONResponse({"message": f"PDF file {filename} has been ingested and split into {len(docs)} documents."})
        except Exception as e:
            return JSONResponse({"message": f"Unable to process the PDF file. Error: {str(e)}"})
        
        # Create  embeddings  and ingest into Pinecone
        try:
            index_name = "rayo-assistive-ai"
            #rayo_index = init_pinecone_index(index_name)
            #vecdb = PineconeVectorStore.from_existing_index(index_name, embed_fn)
            #await vecdb.aadd_documents(docs)
            texts = [doc.page_content for doc in docs]
            metadatas = None
            init_pinecone_index(index_name)
            vecdb = PineconeVectorStore.from_texts(index_name=index_name, texts=texts, 
                                                   embedding=embed_fn, metadatas=metadatas)
            print(vecdb._index.describe_index_stats())
        except Exception as e:
            return JSONResponse({"message": f"Unable to create vectorDB. Error: {str(e)}"})
            
            # Create a chain
        try:
            prompt_str = """
            You are Rayo AI Assistant. Your job is to assist in question-answering tasks.
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know the answer politely. 
            Use four sentences maximum and keep the answer concise.

            Question: {question} 
            Context: {context} 
            Answer:
        """
            chat_prompt = ChatPromptTemplate.from_template(prompt_str)
            num_chunks = 3
            retriever = vecdb.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
            retrieval = RunnableParallel({"question": RunnablePassthrough(), "context": retriever})
            qa_chain = retrieval | chat_prompt | chat_llm 
            return JSONResponse({"message": "PDF file has been ingested and vectorDB has been created."})
        except Exception as e:
                return JSONResponse({"message": f"Unable to create qa chain. Error: {str(e)}"})
    else:
        return JSONResponse({"message": "No content in the pdf file."})

@app.post("/qa_from_pdf")
async def qa_from_pdf(user_query: str ):
    global qa_chain
    if qa_chain is None:
        return JSONResponse({"message": "Please ingest the PDF file first."})
    if user_query is None:
        return JSONResponse({"message": "Query is empty."})
    try:
        out = qa_chain.invoke(user_query)
        print(out)
        return JSONResponse(content={"message": "Response Generated Successfully!", "Response": out.content}, status_code=status.HTTP_200_OK)    
    except Exception as ex:
        return JSONResponse(content={"error": str(ex)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/evaluate_rag")
async def evaluate_rag():
    global qa_chain,vecdb
    if qa_chain is None:
        return JSONResponse({"message": "Please ingest the PDF file 'ParentingEng2402.pdf' first."})
    if vecdb is None:
        return JSONResponse({"message": "Please ingest the PDF file 'ParentingEng2402.pdf' first."})
    
    from datasets import Dataset

    questions= ['what is the significance of the environment a child is exposed to during pregnancy and early childhood',
'describe the cognitive development of a child during early years.',
"How should parents encourage children's speech development ?",
'how can parents effectively handle mischievous behavior in children?']

    ground_truths = ["The environment a child is exposed to during pregnancy and early childhood plays a crucial role in shaping their personality and temperament. The home environment, including relationships with family members, discipline, and exposure to knowledge, significantly influences a child's development. Parents should create a supportive, organized, and disciplined environment at home to foster a love for learning and cultivate positive qualities in children. The behavior, routines, and interactions within the household impact the child's upbringing more than verbal instructions or teachings.",
"During the early years, a child's cognitive development is strongly linked to play and speech. Children under seven years old learn through play and express their thoughts by speaking. Encouraging children to speak and engage in physical activities enhances their ability to think and understand. It is important not to hinder a child's natural curiosity and exploration, as these experiences contribute significantly to their education and development",
"Parents should encourage children's speech development by not stopping them from speaking, providing feedback and counseling in conversations, and creating a conversational atmosphere that boosts their morale. It is important to pay attention to children when they are speaking, encourage them to speak more, and engage in conversations that strengthen their thinking process. Additionally, redirecting a child's attention to more interesting activities rather than scolding them can help in speech development. Lastly, creating a supportive environment and showing gratitude for good qualities can also aid in encouraging children's speech development.",
"Parents can effectively handle mischievous behavior in children by understanding that misbehaving is a fundamental trait of a child. They should engage in conversation with their mischievous child and not discourage them from speaking. It is important to allow children to play, speak, and express themselves as these activities are crucial for their learning and development. Parents should avoid scolding or silencing their children as it can hinder their ability to think and understand."]
    answers = []
    contexts = []
    retriever = vecdb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# Inference
    for query in questions:
        answers.append(qa_chain.invoke(query).content)
        contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

    data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
    dataset = Dataset.from_dict(data)

    result = evaluate(
    dataset = dataset, 
    metrics = [context_precision,context_recall,faithfulness,answer_relevancy,],)

    df = result.to_pandas()
    # write a csv file
    curr_date = str(datetime.now()) 
    csv_filename = "".join(random.choices(string.ascii_letters, k=4)) + curr_date.split('.')[0].replace(':', '-').replace(" ", 'T') + ".csv"
    #csv_file_path=str(Path(__file__).parent.joinpath(csv_filename))
    csv_file_path=str(Path(__file__).parent.joinpath(csv_filename))
    df.to_csv(csv_file_path, index=False)
    return JSONResponse({"message": f"Evaluation completed successfully. Results are saved in the CSV file: {csv_file_path}"})   


if __name__ == "__main__":
    # This line enables listening on all network interfaces.
    # This change is introduces to access running swagger of WSL in windows browser.
    uvicorn.run(app, host="0.0.0.0", port=9090)