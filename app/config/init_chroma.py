import chromadb

def init_chroma():
    # client = chromadb.PersistentClient() # ChromaDB의 영속적 클라이언트를 반환하는 함수
    client = chromadb.Client() # 테스트용 인메모리 방식 클라이언트 생성
    collection = client.collection = client.get_or_create_collection(name="face_recognition")
    return collection