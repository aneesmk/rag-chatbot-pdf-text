# working_rag_chatbot.py - READS YOUR ACTUAL FILES
import os
import sys

print("🚀 RAG Chatbot - Reading YOUR Files")

def load_documents_properly():
    """Load both TXT and PDF files with proper error handling"""
    documents = []
    folder_path = "docs"
    
    print(f"📁 Reading files from: {folder_path}")
    
    if not os.path.exists(folder_path):
        print("❌ 'docs' folder not found! Create a 'docs' folder with your files.")
        return []
    
    # Check what files exist
    files = os.listdir(folder_path)
    print(f"📄 Files found: {files}")
    
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        
        # Handle TEXT files (.txt)
        if filename.endswith('.txt'):
            print(f"🔧 Reading TEXT: {filename}")
            
            try:
                # Try multiple encodings to read the file
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
                
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            content = file.read()
                        
                        # If we get here, reading was successful
                        if content.strip():  # Check if file has content
                            from langchain_core.documents import Document
                            doc = Document(
                                page_content=content,
                                metadata={"source": filename, "type": "text"}
                            )
                            documents.append(doc)
                            print(f"✅ SUCCESS: Loaded {filename} with {encoding} encoding")
                            break
                        else:
                            print(f"⚠️  File {filename} is empty")
                            break
                            
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"❌ Error reading {filename} with {encoding}: {e}")
                        continue
                else:
                    print(f"❌ Failed to read {filename} with any encoding")
                
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")
        
        # Handle PDF files (.pdf)
        elif filename.endswith('.pdf'):
            print(f"🔧 Reading PDF: {filename}")
            
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
                file_docs = loader.load()
                
                # Add source information to each page
                for doc in file_docs:
                    doc.metadata['source'] = filename
                    doc.metadata['type'] = 'pdf'
                    doc.metadata['page'] = doc.metadata.get('page', 0)
                
                documents.extend(file_docs)
                print(f"✅ PDF: Loaded {filename} ({len(file_docs)} pages)")
                
            except Exception as e:
                print(f"❌ PDF Error {filename}: {e}")
        
        else:
            print(f"⚠️  Skipped {filename} (unsupported format)")
    
    print(f"📚 Successfully loaded {len(documents)} document chunks")
    return documents

def main():
    print("=" * 60)
    print("🤖 RAG CHATBOT - USING YOUR FILES")
    print("=" * 60)
    
    # Load YOUR actual files
    documents = load_documents_properly()
    
    if not documents:
        print("\n❌ PROBLEM: No documents could be loaded.")
        print("💡 SOLUTIONS:")
        print("1. Make sure you have a 'docs' folder")
        print("2. Add .txt files to the 'docs' folder") 
        print("3. Check file encoding (try saving as UTF-8)")
        print("4. Ensure files are not empty")
        return
    
    try:
        # Import required modules
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.llms import Ollama
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        
        print(f"\n🔄 Processing {len(documents)} documents...")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Split into {len(chunks)} chunks")
        
        # Create vector database
        print("🔄 Creating search database...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = Chroma.from_documents(chunks, embeddings)
        print("✅ Search database created!")
        
        # Setup AI model
        print("🔄 Loading AI model...")
        llm = Ollama(model="llama3.2:latest")
        print("✅ AI model ready!")
        
        # Create RAG system
        prompt_template = """Use the following context from documents to answer the question. 
        If the answer is not in the context, say "I don't have that information."

        Context: {context}

        Question: {question}
        Answer: """
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=False
        )
        
        # CHAT INTERFACE
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! CHATBOT IS READY!")
        print("=" * 60)
        print(f"📚 Using {len(documents)} of YOUR documents")
        print("💬 Ask questions about your files")
        print("🚪 Type 'exit' to quit")
        print("=" * 60 + "\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if question.lower() in ['exit', 'quit']:
                    print("👋 Goodbye!")
                    break
                    
                if not question:
                    continue
                
                print("🤔 Searching documents...")
                response = rag_chain.invoke({"query": question})
                print(f"🤖 {response['result']}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try another question.\n")
                
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("💡 Check if Ollama is running: 'ollama list'")

if __name__ == "__main__":
    main()
