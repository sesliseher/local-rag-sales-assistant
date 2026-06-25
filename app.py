import os
import json
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 1. VERİLERİ YÜKLEME (DATA INGESTION)
print("1. Belgeler okunuyor...")
docs = []

# CSV Yükleme
csv_loader = CSVLoader(file_path="./satis_verileri/fiyat_listesi.csv", encoding="utf-8")
docs.extend(csv_loader.load())

# TXT Yükleme
txt_loader = TextLoader(file_path="./satis_verileri/garanti_sartlari.txt", encoding="utf-8")
docs.extend(txt_loader.load())

# JSON Yükleme
with open("./satis_verileri/arac_katalogu.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)
    json_text = json.dumps(json_data, ensure_ascii=False, indent=2)
    docs.append(Document(page_content=json_text, metadata={"source": "arac_katalogu.json"}))

# 2. METİNLERİ PARÇALAMA (CHUNKING)
print("2. Metinler vektörleştirme için parçalanıyor...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# 3. VEKTÖRLEŞTİRME VE VERİTABANI (EMBEDDING & CHROMADB)
print("3. Yeni çok dilli vektör veritabanı oluşturuluyor... (İlk seferde modeli indirebilir)")
# İŞTE BURASI YENİ: Türkçeyi anlayan çok dilli vektör modelimiz
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings, persist_directory="./chroma_db")

# 4. LLM VE GERİ ÇAĞIRICI KURULUMU (RETRIEVER)
# İŞTE BURASI YENİ: 3 saat beklediğimiz yeni beynimiz
llm = Ollama(model="llama3")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. PROMPT MÜHENDİSLİĞİ
template = """Sen otomotiv sektöründe çalışan bir satış asistanısın. 
Aşağıdaki bağlam (context) bilgilerini kullanarak müşterinin sorusunu Türkçe cevapla.
Eğer sorunun cevabı bu metinlerde yoksa, sadece 'Bu bilgiye sahip değilim' de.

Bağlam: {context}

Soru: {question}

Cevap:"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# 6. RAG ZİNCİRİNİ OLUŞTURMA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# 7. SİSTEMİ TEST ETME
print("\n" + "="*50)
print("SİSTEM HAZIR! Llama-3 Destekli Satış Asistanı Dinliyor...")
print("(Çıkmak için 'q' yazıp Enter'a basın)")
print("="*50)

while True:
    user_input = input("\nSorunuz: ")
    if user_input.lower() == 'q':
        break

    print("Düşünüyor...")
    response = qa_chain.invoke({"query": user_input})
    
    print("\n[ASİSTAN]:", response["result"])
    print("\n[BAKILAN KAYNAKLAR]:")
    for doc in response["source_documents"]:
        print(f" -> {doc.metadata['source']}")