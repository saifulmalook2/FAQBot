import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import AzureAISearchRetriever




chat_history = {}



async def create_chain(retriever: AzureAISearchRetriever, model):
    system_prompt = "You are an QA chatbot. You are only allowed to use the knowledge base nothing else. Your job is to answer the question based on the knowledge base/documents provided (Always give summarized answers). {context}"

    main_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    retriever_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            (
                "human",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.",
            ),
        ]
    )

    chain = create_stuff_documents_chain(llm=model, prompt=main_prompt)

    # No need to create a separate retriever here; using AzureAISearchRetriever directly
    history_aware_retriever = create_history_aware_retriever(
        llm=model, retriever=retriever, prompt=retriever_prompt
    )

    return create_retrieval_chain(history_aware_retriever, chain)


async def process_chat(chain, question, chat_history):
    # Invoke the chain with input question and chat history
    response = chain.invoke({"input": question, "chat_history": []})
    
    answer = response['answer']

    print("context", response['context'])
    return answer


async def generate_response(uid, q):
    chat_history.setdefault(uid, [])


    retriever = AzureAISearchRetriever(
        api_key=os.getenv("AZURE_SEARCH_KEY"),
        service_name="azure-vector-db",
        index_name="faq-index",
        top_k=6
        )
    # Initialize Azure Chat model
    model = AzureChatOpenAI(
        max_tokens=200,
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    print("got chat")

    # Create chain with Azure Cognitive Search retriever and model
    chain = await create_chain(retriever, model)

    # Process chat with the created chain
    result = await process_chat(chain, q, chat_history[uid])
    
    print(result)
    chat_history[uid].extend(
        [HumanMessage(content=q), AIMessage(content=result)]
    )
       
    return  result.strip()