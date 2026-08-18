from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# Path
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_DIR = BASE_DIR / "knowledge"


# =========================================================
# Config
# =========================================================

ALLOWED_EXTENSIONS = {".txt"}

DEFAULT_TOP_K = 5

MIN_SIMILARITY = 0.01



# =========================================================
# Load Documents
# =========================================================

def load_documents() -> list[dict[str, str]]:
    """
    โหลดไฟล์ .txt ทั้งหมดจาก knowledge/

    รองรับโครงสร้าง:

    knowledge/
        earth/
        water/
        wind/
        fire/
        *.txt


    return:

    [
        {
            document_id:
            filename:
            element:
            text:
        }
    ]

    """

    if not KNOWLEDGE_DIR.exists():

        raise FileNotFoundError(
            f"ไม่พบโฟลเดอร์ knowledge: {KNOWLEDGE_DIR}"
        )


    documents: list[dict[str, str]] = []


    # ค้นทุก folder ย่อย
    for file_path in sorted(
        KNOWLEDGE_DIR.rglob("*")
    ):


        if not file_path.is_file():
            continue


        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue



        text = file_path.read_text(
            encoding="utf-8"
        ).strip()



        if not text:
            continue



        relative_path = file_path.relative_to(
            KNOWLEDGE_DIR
        )


        # หา element จาก folder
        if len(relative_path.parts) > 1:

            element = relative_path.parts[0]

        else:

            element = "system"



        documents.append(
            {

                "document_id":
                    str(relative_path),


                "filename":
                    str(relative_path),


                "element":
                    element,


                "text":
                    text,

            }
        )



    if not documents:

        raise ValueError(
            "ไม่พบไฟล์ .txt ใน knowledge/"
        )


    return documents



# =========================================================
# Chunking
# =========================================================

def split_into_chunks(
    documents: list[dict[str, str]]
) -> list[dict[str, str]]:

    """
    แบ่งเอกสารตามหัวข้อ

    ใช้บรรทัดว่างเป็นตัวแบ่ง

    """

    chunks: list[dict[str, str]] = []



    for document in documents:


        sections = [

            section.strip()

            for section in document["text"].split("\n\n")

            if section.strip()

        ]



        for index, section in enumerate(
            sections,
            start=1
        ):


            chunks.append(
                {


                    "chunk_id":

                        (
                            f"{document['document_id']}"
                            f"_chunk_{index:03d}"
                        ),


                    "document_id":

                        document["document_id"],


                    "filename":

                        document["filename"],


                    "element":

                        document["element"],


                    "text":

                        section,


                }
            )



    if not chunks:

        raise ValueError(
            "ไม่สามารถสร้าง chunk ได้"
        )


    return chunks




# =========================================================
# TF-IDF Retriever
# =========================================================


class TfidfRetriever:
    """
    TF-IDF + Cosine Similarity Retriever

    รองรับภาษาไทยด้วย character n-gram

    """



    def __init__(self) -> None:


        self.documents = load_documents()



        self.chunks = split_into_chunks(
            self.documents
        )



        self.vectorizer = TfidfVectorizer(

            analyzer="char",

            ngram_range=(2, 6),

            lowercase=False,

            sublinear_tf=True,

            norm="l2",

        )



        texts = [

            chunk["text"]

            for chunk in self.chunks

        ]



        self.document_matrix = (

            self.vectorizer.fit_transform(
                texts
            )

        )




    # -----------------------------------------------------

    # Search

    # -----------------------------------------------------


    def search(

        self,

        query: str,

        top_k: int = DEFAULT_TOP_K,

        min_similarity: float = MIN_SIMILARITY,

    ) -> list[dict[str, Any]]:


        query = query.strip()



        if not query:

            return []



        query_vector = (

            self.vectorizer.transform(
                [query]
            )

        )



        similarities = cosine_similarity(

            query_vector,

            self.document_matrix

        )[0]



        ranked_indices = (

            similarities.argsort()[::-1]

        )



        results = []



        for index in ranked_indices:



            score = float(
                similarities[index]
            )



            if score < min_similarity:

                continue



            chunk = self.chunks[index]



            results.append(

                {


                    "rank":

                        len(results)+1,


                    "chunk_id":

                        chunk["chunk_id"],


                    "filename":

                        chunk["filename"],


                    "element":

                        chunk["element"],


                    "text":

                        chunk["text"],


                    "score":

                        round(score,4)

                }

            )



            if len(results) >= top_k:

                break



        return results




# =========================================================
# Build RAG Context
# =========================================================


def build_context(

    results:list[dict[str,Any]]

)->str:


    if not results:

        return ""



    context=[]



    for result in results:


        context.append(

            (

                f"[SOURCE: {result['filename']}]\n"

                f"[ELEMENT: {result['element']}]\n"

                f"{result['text']}"

            )

        )



    return "\n\n".join(context)




# =========================================================
# Terminal Test
# =========================================================


def print_search_results(

    query:str,

    top_k:int=DEFAULT_TOP_K

):


    retriever = TfidfRetriever()



    results = retriever.search(

        query,

        top_k

    )



    print("="*70)

    print("QUERY:")

    print(query)

    print("="*70)



    if not results:


        print(
            "ไม่พบข้อมูล"
        )

        return



    for result in results:


        print()


        print(
            f"อันดับ {result['rank']}"
        )


        print(
            f"ไฟล์: {result['filename']}"
        )


        print(
            f"ธาตุ: {result['element']}"
        )


        print(
            f"คะแนน: {result['score']}"
        )


        print("-"*70)


        print(
            result["text"]
        )


        print("-"*70)





# =========================================================
# Manual Test
# =========================================================


if __name__ == "__main__":


    test_queries = [


        "ธาตุไฟควรกินอาหารอะไร",


        "ธาตุลมมีอาการอะไร",


        "คนธาตุดินควรกินผักอะไร",


        "ธาตุน้ำควรหลีกเลี่ยงอะไร",


        "อาหารรสขมเย็นจืด",

    ]



    for query in test_queries:


        print_search_results(

            query=query,

            top_k=5

        )


        print("\n\n")