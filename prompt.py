from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context","question"],
    template="""Question: {question}
            Context: {context}
            Please provide a concise and accurate answer based on the context provided. 
            If the context does not contain sufficient information to answer the question, respond with "I don't know."
            """
)